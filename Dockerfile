FROM python:3.7
RUN mkdir /poker_ai
ARG LUT_DIR
COPY flop_lossy_2.pkl preflop_lossless.pkl river_lossy_2.pkl turn_lossy_2.pkl /
# Set the environment variable for the tests
ENV LUT_DIR="/" 
# Copy the requirements.
COPY requirements.txt poker_ai/requirements.txt 
# Work from the root of the repo.
WORKDIR /poker_ai
# Install python modules.
RUN pip install -r requirements.txt
# Copy everything else.
COPY . /poker_ai
RUN pip install -e .
