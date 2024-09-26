TARGETS=setup.py requirements.txt Dockerfile MovieDetailgetter.py
IMAGE_NAME="dev-movie-detail-getter"
CURR_DIR=$(shell pwd)

build: $(TARGETS)
	docker build -t $(IMAGE_NAME) . && docker run --rm -i $(IMAGE_NAME)

clean: $(TARGETS)
	docker image rm $(IMAGE_NAME)

test:
	: \
	&& { echo "[DB: DUMP]" && docker run --rm -i -v "${CURR_DIR}:${CURR_DIR}:ro" -w "${CURR_DIR}" $(IMAGE_NAME) db ./input.csv --dump; } \
	&& { echo "[QUERY: rating]" && docker run --rm -i -v "${CURR_DIR}:${CURR_DIR}:ro" -w "${CURR_DIR}" $(IMAGE_NAME) query ./input.csv --movie_title "Planet Earth II"; } \
	&& { echo "[QUERY: funny - year - cowsay]" && docker run --rm -i -v "${CURR_DIR}:${CURR_DIR}:ro" -w "${CURR_DIR}" $(IMAGE_NAME) query ./input.csv --movie_title "Planet Earth II" --get_field year --formatter funny; } \
	&& echo "All DONE !" \
	&& :

