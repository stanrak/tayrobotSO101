"""Subprocess environments with bounded startup waits and worker cleanup."""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _worker


_CLEANUP_ERRORS = (EOFError, OSError, ValueError, AssertionError, RuntimeError)


class ManagedSubprocVecEnv(SubprocVecEnv):
    """Keep SB3's worker protocol while bounding startup IPC and shutdown joins.

    The pinned SB3 constructor receives worker replies without a timeout and
    cannot close a partially constructed instance. Its small startup sequence
    is reproduced here so failed workers are still owned and cleaned up.
    Normal environment stepping continues to use SB3's implementation.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: str = "spawn",
        *,
        startup_timeout: float = 30.0,
        close_timeout: float = 5.0,
        terminate_timeout: float = 1.0,
    ):
        if not env_fns:
            raise ValueError("At least one subprocess environment is required.")
        if any(not math.isfinite(value) or value <= 0 for value in (startup_timeout, terminate_timeout)):
            raise ValueError("Startup and termination timeouts must be finite and positive.")
        if not math.isfinite(close_timeout) or close_timeout < 0:
            raise ValueError("Close timeout must be finite and nonnegative.")
        self.waiting = False
        self.closed = False
        self.remotes = []
        self.work_remotes = []
        self.processes = []
        self.close_timeout = close_timeout
        self.terminate_timeout = terminate_timeout
        self._startup_deadline = time.monotonic() + startup_timeout
        try:
            context = mp.get_context(start_method)
            for env_fn in env_fns:
                remote, work_remote = context.Pipe()
                self.remotes.append(remote)
                self.work_remotes.append(work_remote)
                process = context.Process(
                    target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=True,
                )
                # Retain even a process whose start() fails partway through.
                self.processes.append(process)
                process.start()
                work_remote.close()
            # Check every worker, not only worker zero: VecEnv's render_mode
            # handshake below would otherwise wait forever on another worker.
            for remote in self.remotes:
                remote.send(("get_spaces", None))
            spaces = [self._startup_receive(remote) for remote in self.remotes]
            observation_space, action_space = spaces[0]
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Environment worker cleanup also failed: {cleanup_error}")
            raise
        finally:
            self._startup_deadline = None

    def _startup_receive(self, remote):
        remaining = max(0.0, self._startup_deadline - time.monotonic())
        if not remote.poll(remaining):
            raise TimeoutError("Timed out waiting for subprocess environment initialization.")
        return remote.recv()

    def get_attr(self, attr_name: str, indices=None):
        if self._startup_deadline is None:
            return super().get_attr(attr_name, indices)
        remotes = self._get_target_remotes(indices)
        for remote in remotes:
            remote.send(("get_attr", attr_name))
        return [self._startup_receive(remote) for remote in remotes]

    @staticmethod
    def _is_alive(process) -> bool:
        try:
            return process.is_alive()
        except _CLEANUP_ERRORS:
            return False

    @staticmethod
    def _join_until(processes, timeout: float) -> None:
        # One deadline per phase; shutdown does not multiply the timeout by the
        # number of environments.
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
            except _CLEANUP_ERRORS:
                pass

    def close(self) -> None:
        if self.closed:
            return
        processes = self.processes
        try:
            if not self.waiting:
                for remote in self.remotes:
                    try:
                        remote.send(("close", None))
                    except _CLEANUP_ERRORS:
                        pass
            # Never recv pending steps here: a crashed/hung worker must not
            # prevent shutdown. Closing pipes also releases partial startups.
            for remote in (*self.remotes, *self.work_remotes):
                try:
                    remote.close()
                except _CLEANUP_ERRORS:
                    pass
            self._join_until(processes, self.close_timeout)
            alive = [process for process in processes if self._is_alive(process)]
            for process in alive:
                try:
                    process.terminate()
                except _CLEANUP_ERRORS:
                    pass
            self._join_until(alive, self.terminate_timeout)
            alive = [process for process in alive if self._is_alive(process)]
            for process in alive:
                try:
                    process.kill()
                except (AttributeError, *_CLEANUP_ERRORS):
                    pass
            self._join_until(alive, self.terminate_timeout)
            survivors = [process for process in processes if self._is_alive(process)]
            for process in processes:
                if not self._is_alive(process):
                    try:
                        process.close()
                    except (AttributeError, *_CLEANUP_ERRORS):
                        pass
            if survivors:
                raise RuntimeError(f"Unable to stop {len(survivors)} subprocess environment worker(s).")
        finally:
            self.waiting = False
            self.closed = True
