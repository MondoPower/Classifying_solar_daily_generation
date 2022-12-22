FROM 290556015539.dkr.ecr.ap-southeast-2.amazonaws.com/analytics/base_image:analytics04
ENV PIP_ROOT_USER_ACTION=ignore
# upgrade pip
RUN pip install --upgrade pip
# Get poetry
RUN pip install poetry -q
# Disable virtual environments for Poetry so it installs things right into the container's python
RUN poetry config virtualenvs.create false --local
# Copy the environment definition file and install the environment
COPY pyproject.toml .
# Run the poetry install, this will also do the lock if a lock file is not present
COPY poetry.lock .
RUN poetry install 

#Copy entry point and libaries
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}
COPY src/libs ${LAMBDA_TASK_ROOT}/libs
COPY src/reports ${LAMBDA_TASK_ROOT}/reports
CMD [ "lambda_function.handler" ]