# Continuum
>  Automate Cloud-Edge Infrastructure Deployments and Benchmarks with Continuum. 
> 1. **Infrastructure deployment**: Create and manage virtual compute continuum infrastructure on the cloud or local hardware.
> 2. **Software installation**: Automatically install complex software deployment on the provided infrastructure.
> 3. **Benchmark execution**: Execute application- and system-level benchmarks on the compute continuum deployment.

Quick Jump: [How It Works](#how-it-works) | [Repository Structure](#repository-structure) | [Setup](#setup) | [Installation](#installation) | [Simple Example](#simple-example-15-minutes) | [Figure 6](#figure-6-4-x-15-minutes) | [Figure 7](#figure-7-9-x-15-minutes) | [Figure 8](#figure-8-4-x-15-minutes) | [Further Customization](#further-customization) | [Appendix](#appendix-a-create-an-ubuntu-2004-vm)

<div align="right">
    <img src="./docs/images/architecture.png" width = "30%" align="right">
</div>
<br>

## How It Works
Continuum has the following execution flow:
1. **Infrastructure configuration:** Users define their desired deployment using Continuum's configuration files, which internally get translated to configurations for the infrastructure provider of choice (e.g., QEMU, Google Cloud).
2. **Infrastructure execution:** The provider creates the requested infrastructure with virtual machines and networks.
3. **Software configuration:** Selected software installation scripts are configured and loaded using Ansible.
4. **Software execution:** Ansible playbooks are executed, installing and configuring software for operating services and resource managers on each machine. 
5. **Benchmark configuration** A user-defined benchmark is configured and prepared.
6. **Benchmark execution:** Containerized applications are executed directly (via Docker or Containerd) or via a resource manager (Kubernetes, KubeEdge, etc.). Key metrics are captured, processed, and presented to the user.

## Repository Structure
The Continuum repository has the following structure:

* [application](./application/): Source code of our benchmark applications. The built containers are hosted on DockerHub.
* [benchmark](./benchmark/): Code for the benchmarking setup, execution, and output processing.
* [configuration](./configuration/): Input configuration files for Continuum, including a test framework.
* [configuration_parser](./configuration_parser/): Code for parsing the configuration files.
* [docs](./docs/): Documentation on how to use Continuum.
* [execution_model](./execution_model/): Code for deploying the serverless execution model using OpenFaaS.
* [infrastructure](./infrastructure/): Code for managing infrastructure providers.
* [resource_manager](./resource_manager/): Code for managing distributed services and resource management software (e.g., Kubernetes).
* [scripts](./scripts/): Scripts for replicating paper evaluations.

## Setup (0 - 15 minutes)
For this demo, we will use Continuum with Google Cloud Platform (GCP) as the infrastructure provider of choice.
We use GCP over QEMU, the infrastructure provider used in our CCGRID paper, as it provides similar functionality and benchmark results, but is much easier to set up.
QEMU requires the user to have powerful hardware and perform complex installation and configuration steps.
With GCP, this is handled by the cloud provider instead of the user.
With a GCP Free Trial account, there are no costs of using Continuum with GCP.
Otherwise, the cost of using GCP is limited to several dollars at most.

Continuum with GCP requires the user to have a single computer with internet access.
Continuum has been tested with the Ubuntu 20.04 operating system, and it is highly recommended to use this operating system to replicate our results.
Other operating systems can potentially work as the main software requirements of our framework (Docker, Python, Ansible, Terraform) are available on many operating systems.
However, we can only guarantee successful execution on Ubuntu 20.04.
In the absence of a physical machine with Ubuntu 20.04, a virtual machine can be used.
Any virtual machine provider can be used - we provide an example at the bottom of this README of how to create an Ubuntu 20.04 virtual machine on a physical Ubuntu installation using QEMU.

## Installation (5 minutes)
Install Continuum's software requirements has the following software requirements on Ubuntu 20.04.
We tested with Docker 23.0.0, pip3 20.0.2 for Python 3.8.10, Ansible 2.9.6, and Terraform 1.3.7.

1. Install Docker:
    ```bash
    # 1. Install from the repository:
    # https://docs.docker.com/engine/install/ubuntu/

    # 2. Manage Docker as a non-root user
    # https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user

    # 3. Enable HTTP support
    hostname -I
    # - Select the first IP from this list - this is your machine's IP
    # - Replace the IP_HERE variable with your IP
    sudo mkdir /etc/docker
    sudo touch /etc/docker/daemon.json
    echo '{ "insecure-registries":["IP_HERE:5000"] }' | sudo tee -a /etc/docker/daemon.json
    sudo systemctl restart docker
    ```
2. Install Pip:
    ```bash
    sudo apt install python3-pip
    ```
3. Install Ansible
    ```bash
    sudo apt install ansible
    sudo sed -i '/# command_warnings = False/c\command_warnings = False' /etc/ansible/ansible.cfg
    sudo sed -i '/#callback_whitelist = timer, mail/c\callback_whitelist = profile_tasks' /etc/ansible/ansible.cfg
    ```
4. Install Terraform
    ```bash
    # For more information, see https://developer.hashicorp.com/terraform/cli/install/apt
    sudo apt install terraform
    ```
4. Install Continuum
    ```bash
    git clone https://github.com/atlarge-research/continuum.git
    cd continuum 
    git checkout CCGRID2023-Artifact-Evaluation
    pip3 install -r requirements.txt

    # Make sure the SSH directory and files are there (if not already)
    mkdir ~/.ssh
    touch ~/.ssh/known_hosts
    ```
5. Prepare Continuum for Google Cloud usage
    ```bash
    # 1. Get your projectID from GCP (example: continuum-project-123456)
    # https://cloud.google.com/resource-manager/docs/creating-managing-projects

    # 2. Download your GCP credentials
    # https://cloud.google.com/iam/docs/creating-managing-service-account-keys

    # 3. Update all Continuum's configuration files with your GCP information
    # All parameters are required!
    cd configuration
    python3 gcp_update.py --help
        usage: gcp_update.py [-h] [--region REGION] [--zone ZONE] [--pid PID] [--cred CRED] [--reset]

        optional arguments:
        -h, --help       show this help message and exit
        --region REGION  GCP Region (e.g., europe-west4)
        --zone ZONE      GCP ZONE (e.g., europe-west4-a)
        --pid PID        GCP projectID (e.g., continuum-project-123456)
        --cred CRED      GCP credentials file (e.g., ~/.ssh/continuum-project-123456-123a456bc78d.json)
        --reset          Empty all parameter values

    # Example:
    python3 gcp_update.py \
        --region europe-west4 \
        --zone europe-west4-a \
        --pid continuum-project-123456 \
        --cred ~/.ssh/continuum-project-123456-123a456bc78d.json
    ```

## Simple Example (15 minutes)
First, we execute a simple example to see what Continuum can do:
```bash
cd ..
python3 main.py configuration/gcp_cloud_kubernetes_benchmark.cfg
```
Continuum attempts to start a Kubernetes cluster on Google Cloud, using 2 cloud VMs (one for the Kubernetes control plane, one as a Kubernetes worker) and 1 endpoint VM (the user that offloads data to the cloud).
When Continuum is done, it will output the results of the performed benchmark, see below for an example.

In this example, one endpoint offloaded 5 images to one cloud worker for 300 seconds, where image classification was performed on each image, and the result was sent back to the endpoint. The time between the endpoint application sending the image and the cloud application receiving the image was 107.49 on average, it then took the cloud worker 81.46 ms to process each image on average, for a total end-to-end latency (the time between the endpoint generating an image and getting the image classification output back) of 191.23 ms per image on average.
```bash
------------------------------------
CLOUD OUTPUT
------------------------------------
 worker_id  total_time (s)  delay_avg (ms)  delay_stdev (ms)  proc_time/data (ms)
         0          302.06          107.49             23.92                81.46
------------------------------------
ENDPOINT OUTPUT
------------------------------------
 connected_to  total_time (s)  preproc_time/data (ms)  data_size_avg (kb)  latency_avg (ms)  latency_stdev (ms)
            0          302.45                    0.45               65.28            191.23               24.44
```
Finally, it outputs SSH commands that can be used to SSH into the provided VMs:
```bash
ssh cloud0@34.90.184.203 -i /home/continuum-vm/.ssh/id_rsa_benchmark
ssh cloud1@34.91.43.123 -i /home/continuum-vm/.ssh/id_rsa_benchmark
ssh endpoint0@34.90.242.25 -i /home/continuum-vm/.ssh/id_rsa_benchmark
```
Optionally, you can inspect the Kubernetes cluster running on the cloud VMs:
```bash
# SSH to cloud0 - the Kubernetes control plane
ssh cloud0@34.90.184.203 -i /home/continuum-vm/.ssh/id_rsa_benchmark
kubectl get nodes
kubectl get pods
exit
```

To delete the infrastructure:
```bash
cd ~/.continuum/images
terraform destroy --auto-approve
```

## Figure 6 (4 x 15 minutes)
We reproduce the Figure 6 experiment "Breakdown of the end-to-end latency per deployment" from our CCGRID paper.
The setup differs slightly for GCP compared to QEMU, which we used in the paper: 
First, we use fewer cloud, edge, and endpoint machines to reduce the cost of using GCP. 
Moreover, using many GCP virtual machines requires special permissions, which need to be approved by GCP first. 
Second, we use different resources per VM. 
GCP offers VMs with specific CPU/memory specifications, which we have to adhere to. 
Therefore, the VMs we create have different specifications compared to the experimental setup in our CCGRID paper. 
However, the trend of powerful cloud VMs, less powerful edge VMs, and least powerful endpoint VMs remains.
Third, the GCP VMs are executed on more powerful hardware compared to our local setup used in our CCGRID paper. 
This changes the VM's performance compared to before, however, all VMs are influenced in the same way, making the results still comparable.

Continuum automatically deletes all provided GCP infrastructure at the end of each run - no manual intervention is needed.

To replicate Figure 6 on GCP, using the configurations in `configuration/experiment_large_deployments`:
```bash
# Clean-up old files
cd continuum
rm -r logs/*.log

# Start script
cd scripts
python3 replicate_paper.py Deployments
```

The script will execute all configurations in `configuration/experiment_large_deployments`.
Once it is done, the Figure 6 graph is created in continuum/logs as a PNG file.
The provided infrastructure is automatically deleted at the end of Continuum's execution, unless Continuum crashed.
When starting Continuum again, it will attempt to delete the provided infrastructure of the previous run automatically.

If the script crashes during the execution of one of the 4 experiments (due to any reason), you need to restart the experiment.
The script will re-use the experiments that ran successfully and re-do the crashed experiments.
Do as follows:
```bash
# 1. Remove the log file of the crashed run (most often the last log file)
# - Example:
rm logs/2023-01-17_16:56:04_edge_image_classification.log

# 2. Copy the timestamp of the oldest log file of this run
# - Using the previous example - if your /logs directory looks like this:
# logs/2023-01-17_15:36:12_cloud_image_classification.log
# logs/2023-01-17_15:51:36_cloud_image_classification.log
# logs/2023-01-17_16:24:55_cloud_image_classification.log
#
# - Copy the timestamp 2023-01-17_15:36:12
# - IMPORTANT: The /logs directory should only have log files related to Figure 6 replication, no other log files.
# 
# 3. Restart the experiment
python3 replicate_paper.py -r 2023-01-17_15:36:12 Deployments

# Alternatively, you can delete the entire logs directory and restart everything
rm -r logs/*.log
cd scripts
python3 replicate_paper.py Deployments
```

## Figure 7 (9 x 15 minutes)
The same explanation in Figure 6 applies to Figure 7 - configurations differ slightly between GCP and QEMU, but the results are similar. When the script crashes, see the explanation above on how to restart. To replicate Figure 7, using the configurations in `configuration/experiment_endpoint_scaling`:
```bash
rm -r logs/*.log
cd scripts
python3 replicate_paper.py EndpointScaling
```

## Figure 8 (4 x 15 minutes)
To replicate the performance model heatmap in Figure 8, but with slightly different parameters for GCP (see configuration/model), using the configurations in `configuration/model`:

```bash
rm -r logs/*.log
cd scripts
python3 replicate_model.py 
```

The `replicate_model.py` script uses the same restart logic as `replicate_paper.py`, see the explanation above.

## Further Customization
To further customize deployments with the Continuum framework, you can create your own configuration files.
The configuration template `configuration/template.cfg` lists all configuration options the framework offers, and can function as a good starting point.
Alternatively, you can have a look at and execute all configurations used to test the functionality of the framework, see the code below.
Note that Continuum offers much more functionality than displayed in the CCGRID paper, like bare-metal and serverless deployments.
Some of these deployments require more extensive setups; this is out of the scope of this artifact evaluation.
For more information on how to set up these advanced features, see the main branch of the Continuum framework.
```bash
cd continuum
for i in configuration/tests/terraform/*.cfg; do
    python3 main.py $i || break
done
```

# Appendix A: Create an Ubuntu 20.04 VM
This code snippet shows how to create an Ubuntu 20.04 VM on a physical Ubuntu installation.
```bash
#-------------------------------------
# 1. Install the QEMU stack
sudo apt update 
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils

# Add your user to the libvirt and kvm user group
sudo adduser [username] libvirt
sudo adduser [username] kvm

# Verify that the installation was succesful.
qemu-system-x86_64 --version
sudo systemctl status libvirtd

# If the libvirtd daemon isn't running, do:
sudo systemctl enable --now libvirtd

#-------------------------------------
# 2. Create the Ubuntu 20.04 VM (requirees 20GB in this example - lower can possibly also work)
wget https://releases.ubuntu.com/20.04.3/ubuntu-20.04.3-live-server-amd64.iso
qemu-img create -f qcow2 ubuntu.img 20G

#-------------------------------------
# 3. Boot the VM - example with 4 CPU cores and 8 GB memory (Continuum can work with less)
# On a physical system with a GUI:
sudo qemu-system-x86_64 -hda ubuntu.img --enable-kvm -m 8G -smp 4 -boot d -cdrom ubuntu-20.04.3-live-server-amd64.iso -cpu host -net nic -net user

# On a system without a GUI:
sudo qemu-system-x86_64 -hda ubuntu.img --enable-kvm -m 8G -smp 4 -boot d -cdrom ubuntu-20.04.3-live-server-amd64.iso -cpu host -net nic -net user,hostfwd=tcp::7777-:22
# - Open an SSH session from a machine with a GUI to the GUI-less machine using "ssh -X"
# - Install and run remmina on the GUI-less machine
sudo apt install remmina
remmina
# - This should open a window on your machine with GUI
# - Click on the + icon to create a new connection. Under protocol, select “VNC”, and then under server, add the VNC address displayed in the terminal where you started the VM (for example, 127.0.0.1:5900). Click save and connect to connect to the VM.
#-------------------------------------

# 4. Initialize the VM - don't forget to install an SSH client, and remember your username + password

# 5. Shutdown the VM once it's initialized, and launch again
sudo qemu-system-x86_64 -hda ubuntu.img --enable-kvm -m 8G -smp 4 -cpu host -net nic -net user,hostfwd=tcp::8888-:22 --name ubuntu
# - On a system with GUI: A new screen will open for your VM
# - On a system without GUI (or if you prefer a terminal): Do "ssh [username]@localhost -p 8888"
```