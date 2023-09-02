# Memory usage of containers

## Memory usage of single containers (50 containers)
|   runtime | avg (MB) | max (MB) |
| --------: | -------: | -------: |
|      runc |          |          |
| kata-qemu |     33.5 |    35.70 |
|   kata-fc |     34.9 |    35.85 |

- it is not actually only that
    - that is the container only
- there is more
- vm uses 1.1 GiB on its own


## How many we can deploy
| vm GB |                                       #runc proc | #kata-fc proc | #kata-qemu proc |
| ----: | -----------------------------------------------: | ------------: | --------------: |
|    16 | gets stuck at 108 although mem/cpu far from full |            73 |              78 |
|     8 |                                                  |               |                 |
