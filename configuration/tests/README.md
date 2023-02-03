# Tests
This directory contains a wide variety of configurations that cover almost all of Continuum's features. 
You can run Continuum with all these configurations using the following command:

```bash
for i in configuration/tests/<qemu OR terraform>/*.cfg; do
    python3 main.py $i || break
done
```

**NOTE**: This contains more functionalities than covered in the CCGRID paper.

A run is successful if it prints one or multiple `ssh vm_name@ip_address -i path/to/ssh/key` at the end.
You can also check the log files in `/logs` to validate the runs.

The tests are currently split up per provider, with `qemu` covering local execution, and `terraform` covering execution in the cloud using Google Cloud Platform. 
The latter requires extra configuration, most notably, defining your GCP project name and service key location.
