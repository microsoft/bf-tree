FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
	libssl-dev \
	pkg-config \
	curl \
	llvm \
	llvm-dev \
	git \
	make \
	libxxhash-dev \
	libaio-dev \
	clang

ENV CC=clang
ENV LD=clang

RUN git clone https://github.com/vmware/splinterdb.git \
	&& cd splinterdb \
	&& make \
	&& make install

COPY target/release/benchmark .
COPY benchmark.toml .

ENTRYPOINT ["./benchmark"]
