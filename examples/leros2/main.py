from time import sleep
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.leros2 import env as _env
from lerobot_robot_ure import URe, UReConfig


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 10

    num_episodes: int = 1
    max_episode_steps: int = 1000


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")

    robot = URe(config=UReConfig())

    robot.connect()

    sleep(3)

    runtime = _runtime.Runtime(
        environment=_env.LeRobotEnvironment(robot=robot),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=10,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
