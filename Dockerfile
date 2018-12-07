FROM python:3.6

RUN apt-get update -yq && apt-get install -yq wget unzip vim cmake sudo && \
useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo


# download and install starcraft headless build + maps
RUN wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip && \
unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip -d ~/ && \
wget http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip && \
wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip && \
unzip -P iagreetotheeula Melee.zip -d ~/StarCraftII/Maps && \
unzip -P iagreetotheeula mini_games.zip -d ~/StarCraftII/Maps/ && \
rm *.zip

# copy and install from requirements.txt
COPY . requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
# copy project to container
COPY . /app

ENV C_FORCE_ROOT=true

ENTRYPOINT ["bash"]