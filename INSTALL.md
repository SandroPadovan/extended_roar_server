# Setup ROAR Environment

This file contains instructions on how to set up the C&C server to get it running.
Additionally, some auxiliary scripts are available for advanced usage concerning data analytics or extension and configuration of the C&C server.





## Install Python and Dependencies
First, make sure you have a compatible Python version installed on your system.
All development regarding the server was run and tested with Python 3.10.
Compatibility with other Python versions is not guaranteed.

Second, you need to install the dependencies required by this repository.
As with any Python repository, you'll need pip for that (use the specific version belonging to the Python version you use, e.g., pip 22.3 for Python 3.10).
The list of dependencies is available in [requirements.txt](./requirements.txt).
In order to install, run the following command in the root directory of this repository: `pip install -r requirements.txt`





## Configuration

In the `/config` folder, create a config file `config.ini`. Copy the contents of the `example_config.ini` into
your new config file. You can use this to set various configuration values, without having to modify any of the source
code files.

#### Necessary Config Values

As you can see, many of the values from the `example_config.ini` file are already set to some value. However, you need
to set some important options to start.


| Section       | Key                          | Description                                                                                                      |
|---------------|------------------------------|------------------------------------------------------------------------------------------------------------------|
| `environment` | `clientIP`                   | - IP address of client device, e.g. 192.168.1.100                                                                |
| `filepaths`   | `training_csv_folder_path`   | - **Absolute** path to the training data folder                                                                  |
| `filepaths`   | `evaluation_csv_folder_path` | - **Absolute** path to the evaluation data folder                                                                |
| `filepaths`   | `fingerprints_folder`        | - path to complete set of collected raw_data, used in script to split raw data into training and evaluation sets |
| `filepaths`   | `csv_folder_path`            | - **Absolute** path to fingerprints folder, used in script to split raw data into training and evaluation sets   |

Using the other config options available in the `example_config.ini` file, you can adjust the operation of the
prototypes and auxilary scripts.

For example, by adjusting the `syscall_training_path`, `syscall_test_path`, and `normal_vectorizer_path`,
the data with which the AD is trained can be adjusted, i.e. additional behaviors can be used.




## Fingerprints Folder Structure

All scripts contained in this repository can only work if the required data can be found, i.e., the collected fingerprints need to be stored in a very specific way.

```
FOLDER                                  DESCRIPTION

fingerprints                            # The local folder containing all respective subdirectories. This folder and its children are not required to be located in this repository as long as the corresponding settings are correctly set.
-- evaluation                           # The subfolder where the portion of fingerprints explicitly used only in accuracy computation is stored. The corresponding config option is called `evaluation_csv_folder_path`.
    -- infected-cX                      # Directory for all infected-behavior fingerprints belonging to ransomware configuration X. There should be one folder for every configuration.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- normal                           # Directory for normal-behavior fingerprints. There should be exactly one such folder here.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression-cX                   # Directory for infected compression-behavior fingerprints. There should be one folder for every configuration.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression                      # Directory for compression-behavior fingerprints. There should be exactly one such folder here.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- installation-cX                  # Directory for infected installation-behavior fingerprints. There should be one folder for every configuration.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- installation                     # Directory for installation-behavior fingerprints. There should be exactly one such folder here.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression+installation+normal  # Directory including the concatenation of compression, installation and normal data.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression+normal               # Directory including the concatenation of compression and normal data.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
-- training                             # The subfolder where all other fingerprints used during training will be stored. The corresponding config option is called `training_csv_folder_path`.
    -- infected-cX                      # Directory for all infected-behavior fingerprints belonging to ransomware configuration X. There should be one folder for every configuration.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- normal                           # Directory for normal-behavior fingerprints. There should be exactly one such folder here.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression-cX                   # Directory for infected compression-behavior fingerprints. There should be one folder for every configuration.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression                      # Directory for compression-behavior fingerprints. There should be exactly one such folder here.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- installation-cX                  # Directory for infected installation-behavior fingerprints. There should be one folder for every configuration.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- installation                     # Directory for installation-behavior fingerprints. There should be exactly one such folder here.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression+installation+normal  # Directory including the concatenation of compression, installation and normal data.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
    -- compression+normal               # Directory including the concatenation of compression and normal data.
        -- syscalls                     # Direcotry including the actual syscall fingerprint files.
```





## Setup Complete
All done!
Now you should be able to run the scripts as presented in the [main README](./README.md).
