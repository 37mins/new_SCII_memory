FROM spikingbrain@sha256:3a773ade07540703574ec33410e31f46799814daa65a23ade9510589546c54f9

COPY sc2-qwen-r6-runtime-actual.txt /tmp/sc2-qwen-r6-runtime-actual.txt
RUN ln -s /usr/bin/python3 /usr/local/bin/python \
    && python3 -m pip install --no-cache-dir -r /tmp/sc2-qwen-r6-runtime-actual.txt
