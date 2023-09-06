"""\
Manage applicaiton logic in the framework
Mostly used for calling specific application code
"""

import logging
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

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
        for out in worker_output:
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
    starttime, status = kubernetes.start_worker(config, machines, app_vars, get_starttime=True)

    # Wait for benchmark to finish
    kubernetes.wait_worker_completion(config, machines)

    # Now get raw output
    logging.info("Benchmark has been finished, prepare results")

    worker_output = kubernetes.get_worker_output(config, machines)
    worker_description = kubernetes.get_worker_output(config, machines, get_description=True)

    control_output = kubernetes.get_control_output(config, machines, starttime, status)

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
    )

    if "kata" in config["benchmark"]["runtime"]:
        for ip in config["cloud_ips"]:
            add_kata_timestamps(ip, worker_output)


def gather_kata_traces(ip: str, port: str = "16686") -> List[List[Dict]]:
    """Get jaeger endpoint kata-runtime traces sorted on startTime

    Args:
        ip (str): Jaeger endpoint ip
        port (str): Jaeger endpoint port. Defaults to "16686"

    Returns:
        List[Dict]: A sorted list of traces based on startTime, each sorted by their rootSpans' startTime
    """
    jaeger_api_url = f"http://{ip}:{port}/api/traces?service=kata&operation=rootSpan"
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


def get_kata_period_timestamps(traces: List[List[Dict]]) -> List[List]:
    """For each of the traces, find the periods we are interested in, as below:
    (assuming sorted)
    1. (first) startVM
    2. createContainers
    3. (second) ttrpc.StartContainer

    Args:
        traces (List[List[Dict]]): The list of traces

    Returns:
        List[Tuple[int, int, int]]: A list of lists with the aforementioned periods
    """
    ts = []
    for trace in traces:
        ixs = [
            [i for i, span in enumerate(trace) if span["operationName"] == "startVM"][0],
            next((i for i, d in enumerate(trace) if d["operationName"] == "createContainers"), None),
            [i for i, span in enumerate(trace) if span["operationName"] == "ttrpc.StartContainer"][-1],
        ]
        ts.append([trace[i]["startTime"] for i in ixs])
    return ts


# FIXME: For some numbers (investigate), returns smaller length output and breaks logic
# def iso_time_to_epoch(s):
#     return int(str.replace(f"{datetime.fromisoformat(s[:26] + s[-6:]).timestamp()}", ".", ""))


def _iso_time_to_epoch_subprocess(date: str) -> int:
    cmd = f"date -d '{date}' '+%s%N'"
    out = subprocess.getoutput(cmd)[:-3]
    return int(out)


def _adjust_spans(spans: List[Dict], delta: int) -> List[Dict]:
    return [{k: v + delta if k == "startTime" else v for k, v in span.items()} for span in spans]


def adjust_traces(traces: List[List[Dict]], deltas: List[int]) -> List[List[Dict]]:
    assert len(traces) == len(deltas)
    return [_adjust_spans(trace, delta) for (trace, delta) in zip(traces, deltas)]


def add_kata_timestamps(ip, worker_output):
    print("----------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------")
    print(f"add_kata_timestamps({ip})")

    # skip cache worker
    traces = gather_kata_traces(ip)[1:]
    timestamps = get_kata_period_timestamps(traces)

    sorted_start_app = [str.split(wo[0])[0] for wo in worker_output]
    sorted_start_app_ts = sorted([_iso_time_to_epoch_subprocess(o) for o in sorted_start_app])
    print(sorted_start_app)
    print(sorted_start_app_ts)

    # map every worker's output to it's span - might not be in order
    diff_maps = [{} for _ in range(len(sorted_start_app_ts))]
    for ts_i, ts in enumerate(sorted_start_app_ts):
        diff = float("inf")
        for trace_i, trace in enumerate(traces):
            for span_i, span in enumerate(trace):
                d = ts - span["startTime"]
                if abs(d) < diff:
                    diff = abs(d)
                    diff_maps[ts_i] = {
                        "diff": d,
                        "trace_id": trace_i,
                        "span_id": span_i,
                        "operation": span["operationName"],
                    }

    for i, x in enumerate(diff_maps):
        print(f"i -> {i}")
        print(f"\t{x}")

    deltas = [0] * len(diff_maps)
    for d in diff_maps:
        deltas[d["trace_id"]] = d["diff"]

    adjusted_traces = adjust_traces(traces, deltas)

    adjusted_periods = get_kata_period_timestamps(adjusted_traces)

    files_n = 0
    kata_times = []
