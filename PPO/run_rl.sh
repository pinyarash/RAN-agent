#!/bin/bash
# ./main.py --name=gma_env5_udp --train=1 --env=5 --reward=diff

# ./main.py --name=rl_shift_env5_udp_train --train=1 --env=5 --reward=diff 
# ./main.py --name=rl_env3_udp_train --train=1 --env=3 --reward=diff --shift=False

./main.py --name=gma_env_1 --train=0 --env=1 --reward=diff --load=env5_diff_best --gma=True --trace_time=3 --shift=True 
sleep 10
./main.py --name=gma_env_2 --train=0 --env=2 --reward=diff --load=env5_diff_best --gma=True --trace_time=3 --shift=True 
sleep 10
./main.py --name=gma_env_3 --train=0 --env=3 --reward=diff --load=env5_diff_best --gma=True --trace_time=3 --shift=True 
sleep 10
./main.py --name=gma_env_4 --train=0 --env=4 --reward=diff --load=env5_diff_best --gma=True --trace_time=3 --shift=True 
sleep 10
./main.py --name=gma_env_5 --train=0 --env=5 --reward=diff --load=env5_diff_best --gma=True --trace_time=3 --shift=True 
sleep 10

# ./main.py --name=ppo_env1_udp_pretrain --train=1 --env=1 --reward=diff --shift=True --trace_time=5
# sleep 10
# ./main.py --name=ppo_env2_udp_pretrain --train=1 --env=2 --reward=diff --shift=True --trace_time=5 --load=env1_diff
# sleep 10
# ./main.py --name=ppo_env3_udp_pretrain --train=1 --env=3 --reward=diff --shift=True --trace_time=5 --load=env2_diff
# sleep 10
# ./main.py --name=ppo_env4_udp_pre-train --train=1 --env=4 --reward=diff --shift=True --trace_time=5 --load=env3_diff
# sleep 10
# ./main.py --name=ppo_env5_udp_pre-train --train=1 --env=5 --reward=diff --shift=True --trace_time=5 --load=env4_diff
# sleep 10

# ./main.py --name=ppo_env1_udp_test --train=0 --env=1 --reward=diff --shift=True --trace_time=3 --load=env1_diff
# sleep 10
# ./main.py --name=ppo_env2_udp_test --train=0 --env=2 --reward=diff --shift=True --trace_time=3 --load=env2_diff
# sleep 10
# ./main.py --name=ppo_env3_udp_test --train=0 --env=3 --reward=diff --shift=True --trace_time=3 --load=env3_diff
# sleep 10
# ./main.py --name=ppo_env4_udp_test --train=0 --env=4 --reward=diff --shift=True --trace_time=3 --load=env4_diff
# sleep 10
# ./main.py --name=ppo_env5_udp_test --train=0 --env=5 --reward=diff --shift=True --trace_time=3 --load=env5_diff_best
# sleep 10

# ./main.py --name=ppo_env1_udp_test_seq --train=0 --env=1 --reward=diff --shift=True --trace_time=3 --load=env5_diff_best
# sleep 10
# ./main.py --name=ppo_env2_udp_test_seq --train=0 --env=2 --reward=diff --shift=True --trace_time=3 --load=env5_diff_best
# sleep 10
# ./main.py --name=ppo_env3_udp_test_seq --train=0 --env=3 --reward=diff --shift=True --trace_time=3 --load=env5_diff_best
# sleep 10
# ./main.py --name=ppo_env4_udp_test_seq --train=0 --env=4 --reward=diff --shift=True --trace_time=3 --load=env5_diff_best
# sleep 10
# ./main.py --name=ppo_env5_udp_test_seq --train=0 --env=5 --reward=diff --shift=True --trace_time=3 --load=env5_diff_best
# sleep 10


# ./main.py --name=ppo_env1_udp_con --train=1 --env=1 --reward=diff --shift=False --trace_time=5 --load=pre_train/env4_diff
# sleep 10
# ./main.py --name=ppo_env2_udp_con --train=1 --env=2 --reward=diff --shift=False --trace_time=5 --load=pre_train/env1_diff
# sleep 10
# ./main.py --name=ppo_env3_udp_con --train=1 --env=3 --reward=diff --shift=False --trace_time=5 --load=pre_train/env2_diff
# sleep 10
# ./main.py --name=ppo_env4_udp_con --train=1 --env=4 --reward=diff --shift=False --trace_time=5 --load=pre_train/env3_diff
# sleep 10
# ./main.py --name=ppo_env5_udp_con --train=1 --env=5 --reward=diff --shift=False --trace_time=5 --load=pre_train/env4_diff
# sleep 10


# ./main.py --name=gma_env_all --train=0 --env=0 --reward=diff --load=env5_diff_best --gma=True
# ./main.py --name=rl_env_all_udp_pretrain --train=1 --env=0 --reward=diff --shift=False --load=env0_diff
