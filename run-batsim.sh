docker run \
    --net host \
    -u $(id -u):$(id -g) \
    -v $PWD/simulation:/data \
    oarteam/batsim:3.1.0 \
    -p /data/platform/cluster288-HCMUT-SuperNodeXP.xml \
    -w /data/workload/exact/HCMUT-SuperNodeXP-2017.json \
    --disable-schedule-tracing --disable-machine-state-tracing