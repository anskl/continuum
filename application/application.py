"""\
Manage applicaiton logic in the framework
Mostly used for calling specific application code
"""

import logging
import subprocess
import sys
from datetime import datetime
from statistics import mean
from typing import Dict, List, Tuple

import requests

from execution_model.openfaas import openfaas
from resource_manager.endpoint import endpoint
from resource_manager.kubernetes import kubernetes


def set_container_location(config):
    """[INTERFACE] Set registry location/path of containerized applications

    Args:
        config (dict): Parsed configuration
    """
    config["module"]["application"].set_container_location(config)


def add_options(config):
    """[INTERFACE] Add config options for a particular module

    Args:
        config (ConfigParser): ConfigParser object
    """
    return config["module"]["application"].add_options(config)


def verify_options(parser, config):
    """[INTERFACE] Verify the config from the module's requirements

    Args:
        parser (ArgumentParser): Argparse object
        config (ConfigParser): ConfigParser object
    """
    config["module"]["application"].verify_options(parser, config)


def start(config, machines):
    """[INTERFACE] Start the application with a certain deployment model

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    if config["infrastructure"]["provider"] == "baremetal":
        baremetal(config, machines)
    elif config["benchmark"]["resource_manager"] == "mist":
        mist(config, machines)
    elif config["module"]["execution_model"] and config["execution_model"]["model"] == "openfaas":
        serverless(config, machines)
    elif config["benchmark"]["resource_manager"] == "none":
        endpoint_only(config, machines)
    elif config["benchmark"]["resource_manager"] in ["kubernetes", "kubeedge"]:
        kube(config, machines)
    elif config["benchmark"]["resource_manager"] == "kubecontrol":
        kube_control(config, machines)
    else:
        logging.error("ERROR: Don't have a deployment for this resource manager / application")
        sys.exit()


def print_raw_output(config, worker_output, endpoint_output):
    """Print the raw output

    Args:
        config (dict): Parsed configuration
        worker_output (list(list(str))): Output of each container ran on the edge
        endpoint_output (list(list(str))): Output of each endpoint container
    """
    logging.debug("Print raw output from subscribers and publishers")
    if (config["mode"] == "cloud" or config["mode"] == "edge") and worker_output:
        logging.debug("------------------------------------")
        logging.debug("%s OUTPUT", config["mode"].upper())
        logging.debug("------------------------------------")
        for _, out in worker_output:
            for line in out:
                logging.debug(line)

            logging.debug("------------------------------------")

    if config["infrastructure"]["endpoint_nodes"]:
        logging.debug("------------------------------------")
        logging.debug("ENDPOINT OUTPUT")
        logging.debug("------------------------------------")
        for out in endpoint_output:
            for line in out:
                logging.debug(line)

            logging.debug("------------------------------------")


def to_datetime(s):
    """Parse a datetime string from docker logs to a Python datetime object

    Args:
        s (str): Docker datetime string

    Returns:
        datetime: Python datetime object
    """
    s = s.split(" ")[0]
    s = s.replace("T", " ")
    s = s.replace("+", "")
    s = s[: s.find(".") + 7]
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def baremetal(config, machines):
    """Launch a mist computing deployment

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    # Start the worker
    app_vars = config["module"]["application"].start_worker(config, machines)
    container_names_work = kubernetes.start_worker(config, machines, app_vars)

    # Start the endpoint
    container_names = endpoint.start_endpoint(config, machines)
    endpoint.wait_endpoint_completion(config, machines, config["endpoint_ssh"], container_names)

    # Wait for benchmark to finish
    endpoint.wait_endpoint_completion(config, machines, config["cloud_ssh"], container_names_work)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")
    endpoint_output = endpoint.get_endpoint_output(config, machines, container_names, use_ssh=True)
    worker_output = kubernetes.get_worker_output(config, machines, container_names_work)

    # Parse output into dicts, and print result
    print_raw_output(config, worker_output, endpoint_output)
    worker_metrics = config["module"]["application"].gather_worker_metrics(machines, config, worker_output, None)
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(config, endpoint_output, container_names)
    config["module"]["application"].format_output(config, worker_metrics, endpoint_metrics)


def mist(config, machines):
    """Launch a mist computing deployment

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    # Start the worker
    app_vars = config["module"]["application"].start_worker(config, machines)
    container_names_work = kubernetes.start_worker(config, machines, app_vars)

    # Start the endpoint
    container_names = endpoint.start_endpoint(config, machines)
    endpoint.wait_endpoint_completion(config, machines, config["endpoint_ssh"], container_names)

    # Wait for benchmark to finish
    endpoint.wait_endpoint_completion(config, machines, config["edge_ssh"], container_names_work)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")
    endpoint_output = endpoint.get_endpoint_output(config, machines, container_names, use_ssh=True)
    worker_output = kubernetes.get_worker_output(config, machines, container_names_work)

    # Parse output into dicts, and print result
    print_raw_output(config, worker_output, endpoint_output)
    worker_metrics = config["module"]["application"].gather_worker_metrics(machines, config, worker_output, None)
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(config, endpoint_output, container_names)
    config["module"]["application"].format_output(config, worker_metrics, endpoint_metrics)


def serverless(config, machines):
    """Launch a serverless deployment using Kubernetes + OpenFaaS

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    # Start the worker
    openfaas.start_worker(config, machines)

    # Start the endpoint
    container_names = endpoint.start_endpoint(config, machines)
    endpoint.wait_endpoint_completion(config, machines, config["endpoint_ssh"], container_names)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")
    endpoint_output = endpoint.get_endpoint_output(config, machines, container_names, use_ssh=True)

    # Parse output into dicts, and print result
    print_raw_output(config, None, endpoint_output)
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(config, endpoint_output, container_names)
    config["module"]["application"].format_output(config, None, endpoint_metrics)


def endpoint_only(config, machines):
    """Launch a deployment with only endpoint machines / apps

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    # Start the endpoint
    container_names = endpoint.start_endpoint(config, machines)
    endpoint.wait_endpoint_completion(config, machines, config["endpoint_ssh"], container_names)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")
    endpoint_output = endpoint.get_endpoint_output(config, machines, container_names, use_ssh=True)

    # Parse output into dicts, and print result
    print_raw_output(config, None, endpoint_output)
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(config, endpoint_output, container_names)
    config["module"]["application"].format_output(config, None, endpoint_metrics)


def kube(config, machines):
    """Launch a K8 deployment, benchmarking K8's applications

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    # Cache the worker to prevent loading
    if config["benchmark"]["cache_worker"]:
        app_vars = config["module"]["application"].cache_worker(config, machines)
        kubernetes.cache_worker(config, machines, app_vars)

    # Start the worker
    app_vars = config["module"]["application"].start_worker(config, machines)
    kubernetes.start_worker(config, machines, app_vars)

    # Start the endpoint
    container_names = endpoint.start_endpoint(config, machines)
    endpoint.wait_endpoint_completion(config, machines, config["endpoint_ssh"], container_names)

    # Wait for benchmark to finish
    kubernetes.wait_worker_completion(config, machines)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")
    endpoint_output = endpoint.get_endpoint_output(config, machines, container_names, use_ssh=True)
    worker_output = kubernetes.get_worker_output(config, machines)

    # Parse output into dicts, and print result
    print_raw_output(config, worker_output, endpoint_output)
    worker_metrics = config["module"]["application"].gather_worker_metrics(machines, config, worker_output, None)
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(config, endpoint_output, container_names)
    config["module"]["application"].format_output(config, worker_metrics, endpoint_metrics)


def kube_control(config, machines):
    """Launch a K8 deployment, benchmarking K8's controlplane instead of applications running on it

    Args:
        config (dict): Parsed configuration
        machines (list(Machine object)): List of machine objects representing physical machines
    """
    # Cache the worker to prevent loading
    if config["benchmark"]["cache_worker"]:
        app_vars = config["module"]["application"].cache_worker(config, machines)
        kubernetes.cache_worker(config, machines, app_vars)

    # Start the worker
    app_vars = config["module"]["application"].start_worker(config, machines)
    starttime, kubectl_out, status = kubernetes.start_worker(config, machines, app_vars, get_starttime=True)

    # Wait for benchmark to finish
    kubernetes.wait_worker_completion(config, machines)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")

    worker_output = kubernetes.get_worker_output(config, machines)
    worker_description = kubernetes.get_worker_output(config, machines, get_description=True)

    control_output = kubernetes.get_control_output(config, machines, starttime, status)

    # Add kubectl output
    node = config["cloud_ssh"][0].split("@")[0]
    control_output[node]["kubectl"] = kubectl_out

    kata_traces = get_kata_timestamps(config, worker_output)

    # kata_ts = []
    # if "kata" in config["benchmark"]["runtime"]:
    #     for ip in config["cloud_ips"]:
    #         kata_ts.append(get_kata_timestamps(ip, worker_output))

    # Parse output into dicts, and print result
    print_raw_output(config, worker_output, [])

    config["module"]["application"].format_output(
        config,
        None,
        status=status,
        control=control_output,
        starttime=starttime,
        worker_output=worker_output,
        worker_description=worker_description,
        # kata_ts=[a for b in kata_ts for a in b] if kata_ts is not None else None,
    )


# --------------------------------------------------------------------------------------
# Kata stuff
# --------------------------------------------------------------------------------------


def _gather_kata_traces(ip: str, port: str = "16686") -> List[List[Dict]]:
    """(internal) curl request to jaeger server on `ip` to get the traces produced by the kata runtime.

    Args:
        ip (str): Jaeger endpoint ip
        port (str, optional): Jaeger endpoint port. Defaults to "16686".

    Returns:
        List[List[Dict]]: a sorted list of traces for each kata deployment on `ip`.
    """
    jaeger_api_url = f"http://{ip}:{port}/api/traces?service=kata&operation=rootSpan&limit=10000"
    response = requests.get(jaeger_api_url)
    response_data = response.json()

    traces = response_data["data"]

    # Sort each trace's spans based on starTime and sort traces based on startTime
    traces = sorted(
        [sorted(trace["spans"], key=lambda x: x["startTime"]) for trace in traces],
        key=lambda x: x[0]["startTime"],
    )

    print(f"gather_kata_traces({ip}, {port}) -> got {len(traces)} traces")
    return traces


def get_kata_period_timestamps(traces: List[List[Dict]]) -> Tuple[List[List[int]], List[List[int]]]:
    """For each of the traces, find the periods we are interested in, as below:
    (assuming sorted)
    1. (first) startVM
    2. createContainers
    3. (second) ttrpc.StartContainer

    Args:
        traces (List[List[Dict]]): The list of traces

    Returns:
        Tuple[List[int], List[int]]: A list of indexes (position of period timstamps in span) and their values.
    """
    ix = []
    ts = []
    for trace in traces:
        ixs = [
            [i for i, span in enumerate(trace) if span["operationName"] == "startVM"][0],
            next((i for i, d in enumerate(trace) if d["operationName"] == "createContainers"), None),
            [i for i, span in enumerate(trace) if span["operationName"] == "ttrpc.StartContainer"][-1],
        ]
        ix.append(ixs)
        ts.append([trace[i]["startTime"] for i in ixs])
    return ix, ts


# FIXME: For some numbers (investigate), returns smaller length output and breaks logic
# def iso_time_to_epoch(s):
#     return int(str.replace(f"{datetime.fromisoformat(s[:26] + s[-6:]).timestamp()}", ".", ""))


def _iso_time_to_epoch(date: str) -> int:
    cmd = f"date -d '{date}' '+%s%N'"
    out = subprocess.getoutput(cmd)[:-3]
    return int(out)


def _adjust_spans(spans: List[Dict], delta: int) -> List[Dict]:
    """Adjust `startTime` for each span in trace by adding `delta`.

    Args:
        trace (List[Dict]): A jaeger trace dictionary including `spans`.
        delta (int): The time to add (μs).

    Returns:
        List[Dict]: A new trace with adjusted timestamps.
    """
    return [{k: v + delta if k == "startTime" else v for k, v in span.items()} for span in spans]


def adjust_traces(traces: List[List[Dict]], deltas: Dict) -> List[List[Dict]]:
    """Adjust traces based to their corresponding spans.

    Args:
        traces (List[List[Dict]]): A list of jaeger traces.
        deltas (List[int]): A list of deltas.

    Returns:
        List[List[Dict]]: New list of traces but with adjusted timespans.
    """
    assert len(traces) == len(deltas)
    return [_adjust_spans(trace, deltas[trace[0]["traceID"]]) for trace in traces]

# def get_span_worker_deltas(traces: List[List[Dict]], wo: List[int]) -> List[int]:
#     diff_maps = [{} for _ in range(len(wo))]
#     for ts_i, ts in enumerate(wo):
#         diff = float("inf")
#         for trace_i, trace in enumerate(traces):
#             for span_i, span in enumerate(trace):
#                 d = ts - span["startTime"]
#                 if abs(d) < diff:
#                     diff = abs(d)
#                     diff_maps[ts_i] = {
#                         "diff": d,
#                         "trace_id": trace_i,
#                         "span_i": span_i,
#                         "operation": span["operationName"],
#                     }

#     logging.debug("------------------------------------")
#     logging.debug("for each worker's start print timestamp, print the closest span in the whole trace list")
#     for i, x in enumerate(diff_maps):
#         logging.debug(f"i -> {i}")
#         logging.debug(f"\t{x}")
#     logging.debug("------------------------------------")

#     deltas = [0] * len(diff_maps)
#     for d in diff_maps:
#         deltas[d["trace_id"]] = d["diff"]

#     return deltas

def get_deltas_kata(wo: List[int], traces: List[List[Dict]], kata_end_span_ix: List[int]) -> Dict:
    # span_ID = [trace[ix]["traceID"] for (trace, ix) in zip(traces, kata_end_span_ix)]
    kata_end_span_startTime = [trace[ix]["startTime"] for (trace, ix) in zip(traces, kata_end_span_ix)]

    pairs = [(a, b) for a in wo for b in kata_end_span_startTime]
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[0] - x[1]))

    matched = set()
    min_diff = []
    for a, b in sorted_pairs:
        if a not in matched and b not in matched:
            min_diff.append((a, b))
            matched.add(a)
            matched.add(b)

    print("--------------------------------------------------------------------------------")
    print("adjust_traces_worker_output")
    print(f"avg diff -> {mean(abs(a-b) for (a,b) in min_diff):>{10},}")
    print("--------------------------------------------------------------------------------")

    # for r in min_diff:
    #     print(r)
    # print("--------------------------------------------------------------------------------")

    deltas = {}
    for w, span_t in min_diff:
        trace_id = [a for b in traces for a in b if a["startTime"] == span_t][0]["traceID"]
        deltas[trace_id] = w - span_t

    return deltas


# Kata entry point.
def get_kata_timestamps(config, worker_output) -> List[List[int]]:
    logging.info("----------------------------------------------------------------------------------------")
    logging.info("get_kata_timestamps")
    logging.info("----------------------------------------------------------------------------------------")

    nodes_names, nodes_ips = map(list, zip(*[str.split(x, "@") for x in config["cloud_ssh"][1:]]))

    # won't use for now
    # all_nodes_kata_traces = [_gather_kata_traces(ip)[1:] for ip in nodes_ips]

    traces = _gather_kata_traces(nodes_ips[0])[1:]
    kata_p_ix, _ = get_kata_period_timestamps(traces)

    worker_output_start = [str.split(wo[1][0])[0] for wo in worker_output]
    worker_output_start_epoch_sorted = sorted([_iso_time_to_epoch(o) for o in worker_output_start])

    # logging.debug("------------------------------------")
    # logging.debug(worker_output_start)
    # logging.debug("--> converted to -->")
    # logging.debug(worker_output_start_epoch_sorted)
    # logging.debug("------------------------------------")

    deltas = get_deltas_kata(worker_output_start_epoch_sorted, traces, [x[-1] for x in kata_p_ix])

    adjusted_traces = adjust_traces(traces, deltas)

    assert all(
        x == adjusted_traces[i][kata_p_ix[i][2]]["startTime"] for i, x in enumerate(worker_output_start_epoch_sorted)
    ), "adjusted correctly"

    kata_timestamps = get_kata_period_timestamps(adjusted_traces)[1]
    logging.debug("------------------------------------")
    logging.debug("kata timestamps")
    logging.debug(kata_timestamps)
    logging.debug("------------------------------------")
    return kata_timestamps
