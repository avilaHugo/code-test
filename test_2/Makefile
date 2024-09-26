TARGETS=requirements.txt Dockerfile part2_solver.py
IMAGE_NAME=part-2-solver
CURR_DIR=$(shell pwd)

build: $(TARGETS)
	docker build -t $(IMAGE_NAME) . && docker run --rm -i $(IMAGE_NAME)

clean: $(TARGETS)
	docker image rm $(IMAGE_NAME)

test: input_1.tsv input_2.tsv expected_output.tsv
	diff \
		<(sort expected_output.tsv) \
		<(docker run --rm -i -v "${CURR_DIR}:${CURR_DIR}:ro" -w "${CURR_DIR}" $(IMAGE_NAME) input_1.tsv input_2.tsv | sort) \
&& echo "Match !"

