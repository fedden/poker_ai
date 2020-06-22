FROM pokerai/pokerai:1.0.0rc1
# Copy the requirements.
COPY requirements.txt requirements.txt 
# Install python modules.
RUN pip install -r requirements.txt
# Copy everything else.
COPY . /poker_ai
RUN pip install -e .
# Setup tests.
RUN pip install pytest-cov
ENV CC_TEST_REPORTER_ID=607f73633cb88df8c21568f855bd394dc47772d2228b2f0476ad8c87b625a334 
RUN curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > ./cc-test-reporter 
RUN chmod +x ./cc-test-reporter
RUN ./cc-test-reporter before-build
CMD ["/bin/bash"]
