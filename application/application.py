"""\
Manage applicaiton logic in the framework
Mostly used for calling specific application code
"""

import logging
import sys

from datetime import datetime

from resource_manager.kubernetes import kubernetes
from resource_manager.endpoint import endpoint
from execution_model.openfaas import openfaas


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
    worker_metrics = config["module"]["application"].gather_worker_metrics(
        machines, config, worker_output, None
    )
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(
        config, endpoint_output, container_names
    )
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
    worker_metrics = config["module"]["application"].gather_worker_metrics(
        machines, config, worker_output, None
    )
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(
        config, endpoint_output, container_names
    )
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
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(
        config, endpoint_output, container_names
    )
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
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(
        config, endpoint_output, container_names
    )
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
    worker_metrics = config["module"]["application"].gather_worker_metrics(
        machines, config, worker_output, None
    )
    endpoint_metrics = config["module"]["application"].gather_endpoint_metrics(
        config, endpoint_output, container_names
    )
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

    for ip in config['cloud_ips']:
        get_kata_timestamps(ip)


def get_kata_timestamps(ip):
    # import os
    # import subprocess
    # import json
    import requests
    from statistics import mean

    jaeger_api_url = f"http://{ip}:16686/api/traces?service=kata&operation=rootSpan"
    response = requests.get(jaeger_api_url)
    response_data = response.json()

    # output_dir = "trace_data"
    # subprocess.run("rm -rf trace_data/*", shell=True)
    # os.makedirs(output_dir, exist_ok=True)

    # The cache worker should be skipped
    # -> is the first one to be started
    cache_worker_traceID = sorted(
        [span for trace in response_data["data"] for span in trace["spans"] if span["operationName"] == "rootSpan"],
        key=lambda x: x["startTime"],
    )[0]["traceID"]

    print("----------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------")
    print('DRY RUN get_kata_timestamps')

    files_n = 0
    kata_t = []
    createSandbox_t = []
    createNetwork_t = []
    startVM_t = []
    createContainers_t = []

    for trace in response_data["data"]:
        # print("----------------------------------------------------------------------------------------")
        traceID = trace["traceID"]

        if traceID == cache_worker_traceID:
            continue

        files_n = files_n + 1

        # sort spans in trace based on startTime
        trace = sorted(trace["spans"], key=lambda x: x["startTime"])

        assert len([span for span in trace if span["operationName"] == "rootSpan"]) == 1, "only one rootspan"
        assert trace[1]["operationName"] == "create"

        createSandboxFromConfig_span = [span for span in trace if span["operationName"] == "createSandboxFromConfig"]
        assert len(createSandboxFromConfig_span) == 1
        createSandboxFromConfig_span_id = createSandboxFromConfig_span[0]["spanID"]
        createSandboxFromConfig_span_children = [
            span
            for span in trace
            if span.get("references") and span["references"][0]["spanID"] == createSandboxFromConfig_span_id
        ]

        assert createSandboxFromConfig_span_children[0]["operationName"] == "createSandbox"
        assert createSandboxFromConfig_span_children[1]["operationName"] == "createNetwork"
        assert createSandboxFromConfig_span_children[2]["operationName"] == "startVM"
        # assert createSandboxFromConfig_span_children[3]["operationName"] == "ttrpc.GetGuestDetails"
        assert createSandboxFromConfig_span_children[4]["operationName"] == "createContainers"

        kata_t.append(trace[1]["duration"])
        createSandbox_t.append(createSandboxFromConfig_span_children[0]['duration'])
        createNetwork_t.append(createSandboxFromConfig_span_children[1]['duration'])
        startVM_t.append(createSandboxFromConfig_span_children[2]['duration'])
        createContainers_t.append(createSandboxFromConfig_span_children[4]['duration'])

        print(f"traceID ->            {traceID}")
        print(f"   Kata took in total          -> {trace[1]['duration']:>{13},} μs")
        print(f"1. createSandbox               -> {createSandbox_t[-1]:>{13},} μs")
        print(f"2. createNetwork               -> {createNetwork_t[-1]:>{13},} μs")
        print(f"3. startVM                     -> {startVM_t[-1]:>{13},} μs")
        print(f"4. createContainers            -> {createContainers_t[-1]:>{13},} μs")

        print("------------------------------------------------------------------------")

        # trace_filename = os.path.join(output_dir, f"trace_{trace[0]['startTime']}_{traceID}.json")
        # with open(trace_filename, "w") as f:
        #     json.dump(trace, f, indent=2)

    print(f"checked {files_n} files")
    print(f"average total kata time        -> {int(mean(kata_t)):>{13},} μs (10^-6s)")
    print(f"average createSandbox time     -> {int(mean(createSandbox_t)):>{13},} μs (10^-6s)")
    print(f"average createNetwork time     -> {int(mean(createNetwork_t)):>{13},} μs (10^-6s)")
    print(f"average startVM time           -> {int(mean(startVM_t)):>{13},} μs (10^-6s)")
    print(f"average createContaienrs time  -> {int(mean(createContainers_t)):>{13},} μs (10^-6s)")
