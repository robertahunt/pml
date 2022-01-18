FROM nvidia/cuda:11.3.0-base-ubuntu20.04 AS base

# necessary to be able to save files as 'rob' user instead of root
ARG USERNAME=rob
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN apt-get update
RUN apt-get install nano git openssh-client -y 
#RUN apt-get install python3 pip


RUN mkdir -p /home/rob/.ssh 
RUN chown -R rob:rob /home/rob/.ssh
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /home/rob/.ssh/config

# Install necessary python packages
#RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# For type checking
#RUN pip3 install types-six types-requests pandas

# For running jupyter notebooks
#RUN pip3 install jupyter
ENV JUPYTER_TOKEN="easy"

# CMD needs to be run apparently so that everything afterwards can access the volumes 
CMD ["/bin/bash"]


# In case it already exists, remove the old git folder
#RUN rm -rf $REPO_FOLDER
RUN export PATH=$PATH:/home/rob/.local/bin
#RUN python3 -m pip install black pytest gitpython

RUN apt-get install nano git openssh-client openssl make gcc libssl-dev wget  libbz2-dev zlib1g-dev build-essential -y
ENV TZ="Africa/Lusaka"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime  && echo $TZ > /etc/timezone
RUN apt-get install tzdata -y
RUN apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 liblzma-dev  -y
RUN apt-get install libsqlite3-dev sqlite3


RUN wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz -P /opt
RUN tar xzvf /opt/Python-3.6.8.tgz -C /opt
RUN cd /opt/Python-3.6.8
RUN ./opt/Python-3.6.8/configure
RUN make
RUN make install
RUN ln -fs /opt/Python-3.6.8/Python /usr/bin/python3.6

RUN python3.6 -m pip install --upgrade setuptools pip
RUN python3.6 -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# used to speed up adding changes to the build process. Everything above should be pretty static
FROM base AS build1

ENV WORLD_VIEW="DOCKER"
RUN python3.6 -m pip install black pytest gitpython pandas types-six types-requests tqdm scikit-image imutils
RUN python3.6 -m pip install opencv-python albumentations bokeh scikit-learn plotly
# used to calculate the s-dbw cluster similarity metric
RUN python3.6 -m pip install s-dbw 
RUN python3.6 -m pip install biopython seaborn prettytable

RUN python3.6 -m pip freeze > /home/rob/requirements.txt

RUN mkdir /home/ngw861
RUN mkdir /home/ngw861/pml
RUN chown rob: /home/ngw861 -R

# Switch to rob user
USER rob