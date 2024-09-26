FROM python:3.12-alpine AS base_image

ARG VENV="/opt/venv"

WORKDIR /tmp/build

RUN : \
    && apk update \
    && apk add --no-cache git \
    && :

COPY requirements.txt requirements.txt

RUN : \
    && /usr/bin/env python3 -m venv "${VENV}" \
    && "${VENV}/bin/python3" -m pip install --upgrade --no-cache-dir pip \
    && "${VENV}/bin/python3" -m pip install --no-cache-dir -r requirements.txt \
    && :

COPY setup.py setup.py
COPY MovieDetailgetter.py MovieDetailgetter.py

RUN : \
    && "${VENV}/bin/python3" setup.py install \
    && :

FROM python:3.12-alpine AS runner

ARG VENV="/opt/venv"

COPY --from=base_image "${VENV}" "${VENV}"

ENV PATH="${VENV}/bin:${PATH}"

ENTRYPOINT [ "MovieDetailgetter.py" ]
CMD [ "--help" ]
