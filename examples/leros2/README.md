
To start the inference server run:

```bash
SERVER_ARGS="--env LEROS2 --default_prompt='put the red cube on the black square'" docker compose -f examples/leros2/compose.yml up   
```

After this you can source your ROS 2 environment and run:

```bash
uv run examples/leros2/main.py
```