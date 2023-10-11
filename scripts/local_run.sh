#!/usr/bin/env bash
trap 'kill 0' SIGINT

python projects/harobo/eval_baselines_agent.py \
    --num_episodes 100 \
    --baseline_config_path projects/harobo/configs/agent/harobo_agent_place_hr.yaml \
    --env_config_path projects/harobo/configs/env/harobo_eval_place_hr.yaml &

# python projects/harobo/eval_baselines_agent.py \
#     --num_episodes 100 \
#     --baseline_config_path projects/harobo/configs/agent/harobo_agent_place_rl.yaml \
#     --env_config_path projects/harobo/configs/env/harobo_eval_place_rl.yaml &

wait