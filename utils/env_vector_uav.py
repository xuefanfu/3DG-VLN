import signal
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from threading import Thread
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
import numpy as np
import attr
import copy

from src.common.param import args

from utils.pickle5_multiprocessing import ConnectionWrapper
from utils.env_utils_uav import ENV
from utils.logger import logger

COMMAND_CLOSE = "close"
COMMAND_SET_BATCH = "set_batch"
COMMAND_GET_OBS = "get_obs_at"
COMMAND_GET_COLLISION_SENSOR = 'get_collision_sensor'


try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch
    from torch import multiprocessing as mp  # type:ignore
except ImportError:
    torch = None
    import multiprocessing as mp  # type:ignore


@attr.s(auto_attribs=True, slots=True)
class _ReadWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    should have something to read.
    """
    read_fn: Callable[[], Any]
    rank: int
    is_waiting: bool = False

    def __call__(self) -> Any:
        if not self.is_waiting:
            raise RuntimeError(
                f"Tried to read from process {self.rank}"
                " but there is nothing waiting to be read"
            )
        res = self.read_fn()
        self.is_waiting = False

        return res


@attr.s(auto_attribs=True, slots=True)
class _WriteWrapper:
    r"""Convenience wrapper to track if a connection to a worker process
    can be written to safely.  In other words, checks to make sure the
    result returned from the last write was read.
    """
    write_fn: Callable[[Any], None]
    read_wrapper: _ReadWrapper

    def __call__(self, data: Any) -> None:
        if self.read_wrapper.is_waiting:
            raise RuntimeError(
                f"Tried to write to process {self.read_wrapper.rank}"
                " but the last write has not been read"
            )
        self.write_fn(data)
        self.read_wrapper.is_waiting = True


class VectorEnvUtil:

    _num_envs: int
    _mp_ctx: BaseContext
    _workers: List[Union[mp.Process, Thread]]
    _connection_read_fns: List[_ReadWrapper]
    _connection_write_fns: List[_WriteWrapper]

    def __init__(
        self,
        load_scenes,
        num_envs: int = 1,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ) -> None:
        """..

        :param make_env_fn: function which creates a single environment. An
            environment can be of type :ref:`env.Env` or :ref:`env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            :ref:`_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU useage.
        :param workers_ignore_signals: Whether or not workers will ignore SIGINT and SIGTERM
            and instead will only exit when :ref:`close` is called
        """
        self._is_closed = True

        self.load_scenes = load_scenes
        self._num_envs = int(num_envs)

        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args={
                'load_scenes': load_scenes,
            },
            workers_ignore_signals=workers_ignore_signals,
        )

        self._is_closed = False

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn_args: dict,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = ENV(
            load_scenes=env_fn_args['load_scenes'],
        )

        if parent_pipe is not None:
            parent_pipe.close()

        try:
            command, data = connection_read_fn()
            while command != COMMAND_CLOSE:
                if command == COMMAND_SET_BATCH:
                    env.set_batch(data)
                    connection_write_fn(True)

                elif command == COMMAND_GET_OBS:
                    index, state = data
                    (teacher_action, done, oracle_success), state = env.get_obs_at(index, state)
                    connection_write_fn(
                        ((teacher_action, done, oracle_success), state)
                    )

                elif command == COMMAND_GET_COLLISION_SENSOR:
                    index, state = data
                    is_collision = env.get_collision_sensor_result_at(index, state)
                    connection_write_fn(bool(is_collision))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()
        except KeyboardInterrupt:
            print("Worker KeyboardInterrupt")
        except Exception as e:
            logger.error(e)
            try:
                logger.error('command is: {} \t data is: {}'.format(command, data))
            except:
                pass
        finally:
            if child_pipe is not None:
                child_pipe.close()

    def _spawn_workers(
        self,
        env_fn_args,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        parent_connections, worker_connections = zip(
            *[
                [ConnectionWrapper(c) for c in self._mp_ctx.Pipe(duplex=True)]
                for _ in range(self._num_envs)
            ]
        )
        self._workers = []
        for worker_conn, parent_conn in zip(worker_connections, parent_connections):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    env_fn_args,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()

        read_fns = [
            _ReadWrapper(p.recv, rank)
            for rank, p in enumerate(parent_connections)
        ]
        write_fns = [
            _WriteWrapper(p.send, read_fn)
            for p, read_fn in zip(parent_connections, read_fns)
        ]

        return read_fns, write_fns

    def close(self) -> None:
        if self._is_closed:
            return

        for read_fn in self._connection_read_fns:
            if read_fn.is_waiting:
                read_fn()
                
        for write_fn in self._connection_write_fns:
            write_fn((COMMAND_CLOSE, ''))

        for process in self._workers:
            process.join()

        self._is_closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)

        for index in range(self._num_envs):
            self._connection_write_fns[index](
                (COMMAND_SET_BATCH, copy.deepcopy(batch))
            )

        results = [
            self._connection_read_fns[index]() for index in range(self._num_envs)
        ]

        return


    def get_obs(self, obs_states) -> Tuple[List[Any], List[Any]]:
        self.obs_states = obs_states

        for index in range(len(obs_states)):
            _, _, state, _, _ = obs_states[index]
            self._connection_write_fns[index](
                (COMMAND_GET_OBS, (index, state))
            )

        results = [
            self._connection_read_fns[index]() for index in range(len(obs_states))
        ]

        obs = []
        sim_states = []
        for index in range(len(obs_states)):
            (teacher_action, done, oracle_success), sim_state = results[index]
            self.obs_states[index] = (obs_states[index][0], obs_states[index][1], sim_state, obs_states[index][3], obs_states[index][4])

            obs.append(
                self._format_obs_at(index, teacher_action, done, oracle_success)
            )
            sim_states.append(sim_state)

        return obs, sim_states

    # def _format_obs_at(self, index: int, teacher_waypoints, done, oracle_success):
    #     rgb_images, depth_images, sim_state, rgb_records, depth_records = self.obs_states[index]
    #     observations = [info for info in sim_state.trajectory[-5:]]
    #     observations[-1]['instruction'] = sim_state.raw_trajectory_info['instruction']
    #     observations[-1]['trajectory_dir'] = sim_state.raw_trajectory_info['trajectory_dir']
    #     observations[-1]['teacher_action'] = teacher_waypoints
    #     observations[-1]['rgb'] = rgb_images
    #     observations[-1]['depth'] = depth_images
    #     observations[-1]['rgb_record'] = rgb_records
    #     observations[-1]['depth_record'] = depth_records
    #     collision = sim_state.is_collisioned

    #     return observations, done, collision, oracle_success
# state_info_results [[{'sensors': {'state': {'position': [141.22314453125, 23.355615615844727, -11.663724899291992], 'linear_velocity': [-4.3181065848330036e-05, 1.4748272405995522e-05, 0.11753630638122559], 'linear_acceleration': [-0.0032405282836407423, 0.0011002026731148362, 8.704916954040527], 'orientation': [0.000541645276825875, -0.001460603903979063, 0.6300283074378967, -0.7765705585479736], 'angular_velocity': [-0.0009696830529719591, -0.0001732670934870839, -6.2396527944486024e-09], 'angular_acceleration': [1.4450292587280273, -0.6892037391662598, -7.66313306144184e-08], 'collision': {'has_collided': False, 'object_name': ''}, 'gps_location': [47.6427381553407, -122.13985417487297, 133.66574096679688], 'timestamp': 1766564531233234944}, 'imu': {'time_stamp': 1766564531233234944, 'rotation': [[0.20612439692634954, 0.978521286957752, 0.002951027693041418], [-0.9785244514745757, 0.2061280768946664, -0.0009991920606031435], [-0.0015860202650800695, -0.00268169516124131, 0.9999951465132595]], 'orientation': [0.000541645276825875, -0.001460603903979063, 0.6300283074378967, -0.7765705585479736], 'linear_acceleration': [-0.12202036380767822, -0.0004880751948803663, -1.1359397172927856], 'angular_velocity': [-0.0010244715958833694, -0.0026171375066041946, 0.0014803928788751364]}}}, {'sensors': {'state': {'position': [22.59379768371582, 325.9439697265625, -9.414101600646973], 'linear_velocity': [-3.304109486634843e-05, 3.8192338251974434e-05, 0.11753310263156891], 'linear_acceleration': [-0.002396480878815055, 0.002777581103146076, 8.704679489135742], 'orientation': [9.467677591601387e-05, -0.0016298460541293025, 0.794521152973175, -0.6072343587875366], 'angular_velocity': [0.020458554849028587, -0.011950301006436348, 7.016677017190887e-08], 'angular_acceleration': [1.3558298349380493, -0.7997112274169922, 1.828805761761032e-05], 'collision': {'has_collided': False, 'object_name': ''}, 'gps_location': [47.641671125873174, -122.13582730451219, 131.4228515625], 'timestamp': 1766564532407562496}, 'imu': {'time_stamp': 1766564532407562496, 'rotation': [[-0.26253303783996706, 0.9649207771204612, 0.002129842449524356], [-0.9649213943547398, -0.26252774297103065, -0.0024749123495603226], [-0.0018289516448820589, -0.0027048763148220505, 0.9999946692762959]], 'orientation': [9.467677591601387e-05, -0.0016298460541293025, 0.794521152973175, -0.6072343587875366], 'linear_acceleration': [0.008507763966917992, 0.08525960147380829, -1.1450543403625488], 'angular_velocity': [0.01901715248823166, -0.014288634061813354, 0.0014572282088920474]}}}]]

    def _format_obs_at(self, index: int, teacher_waypoints, done, oracle_success):
        rgb_images, depth_images, sim_state, rgb_records, depth_records = self.obs_states[index]
        observations = [info for info in sim_state.trajectory[-1:]]
        observations[-1]['instruction'] = sim_state.raw_trajectory_info['instruction']
        observations[-1]['trajectory_dir'] = sim_state.raw_trajectory_info['trajectory_dir']
        observations[-1]['teacher_action'] = teacher_waypoints
        observations[-1]['rgb'] = rgb_images
        observations[-1]['depth'] = depth_images
        observations[-1]['rgb_record'] = rgb_records
        observations[-1]['depth_record'] = depth_records
        collision = sim_state.is_collisioned

        return observations, done, collision, oracle_success


    def get_collision_sensor(self, states) -> List[Any]:
        for index in range(len(states)):
            self._connection_write_fns[index](
                (COMMAND_GET_COLLISION_SENSOR, (index, states[index]))
            )

        results = [
            self._connection_read_fns[index]() for index in range(len(states))
        ]

        return results

