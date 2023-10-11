FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023

# install baseline agent requirements
RUN /bin/bash -c "\
    . activate home-robot \
    && cd home-robot \
    && git submodule update --init --recursive src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet \
    && pip install -e src/third_party/detectron2 \
    && pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt \
    && pip install -e src/home_robot \
    "

# download pretrained Detic checkpoint
RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate

# download pretrained skills
RUN /bin/bash -c "\
    mkdir -p home-robot/data/checkpoints \
    && cd home-robot/data/checkpoints \
    && wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023.zip \
        -O ovmm_baseline_home_robot_challenge_2023.zip \
        --no-check-certificate \
    && unzip ovmm_baseline_home_robot_challenge_2023.zip \
    && rm ovmm_baseline_home_robot_challenge_2023.zip \
    "

ADD igp /home-robot/data/checkpoints/igp
# add harobo agent code
ADD harobo /home-robot/projects/harobo/harobo
ADD harobo/perception/detection/detic/detic_perception_harobo.py /home-robot/src/home_robot/home_robot/perception/detection/detic/detic_perception_harobo.py
ADD harobo/perception/detection/grounded_sam/grounded_sam_harobo.py /home-robot/src/home_robot/home_robot/perception/detection/grounded_sam/grounded_sam_harobo.py
ADD configs /home-robot/projects/harobo/configs

# add eval code
ADD eval_baselines_agent_rl.py /home-robot/projects/harobo/agent.py
ADD evaluator.py /home-robot/projects/harobo/evaluator.py
ADD utils /home-robot/projects/harobo/utils

# add submission script
ADD scripts/submission.sh /home-robot/submission.sh

# set evaluation type to remote
ENV AGENT_EVALUATION_TYPE remote

# additional command line arguments for local evaluations (empty for remote evaluation)
ENV LOCAL_ARGS ""

# run submission script
CMD /bin/bash -c "\
    . activate home-robot \
    && cd /home-robot \
    && export PYTHONPATH=/evalai_remote_evaluation:$PYTHONPATH \
    && bash submission.sh $LOCAL_ARGS \
    "
