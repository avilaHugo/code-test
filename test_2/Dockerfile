FROM python:3.12-alpine AS base_image

ARG VENV="/opt/venv"

WORKDIR /tmp/build

COPY requirements.txt requirements.txt

RUN : \
    && /usr/bin/env python3 -m venv "${VENV}" \
    && "${VENV}/bin/python3" -m pip install --upgrade --no-cache-dir pip \
    && "${VENV}/bin/python3" -m pip install --no-cache-dir -r requirements.txt \
    && :

COPY setup.py setup.py
COPY part2_solver.py part2_solver.py

RUN : \
    && "${VENV}/bin/python3" setup.py install \
    && :

FROM python:3.12-alpine AS runner

ARG VENV="/opt/venv"

COPY --from=base_image "${VENV}" "${VENV}"

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

USER appuser

ENV PATH="${VENV}/bin:${PATH}"

ENTRYPOINT [ "part2_solver.py" ]
CMD [ "--help" ]


