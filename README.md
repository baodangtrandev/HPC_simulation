# HPC Simulation with Batsim and HDEM-based Backfilling

## **Project Introduction:**

This project focuses on simulating a High Performance Computing (HPC) environment using **Batsim**. Additionally, it integrates the HDEM prediction model to enhance the efficiency of scheduling algorithms, especially the backfilling method. The goal is to provide users with a platform to experiment with and evaluate advanced scheduling techniques in a realistic simulation environment.

---

## **System Requirements:**

To use and run this codebase, you should prepare the following components:

- **Python 3.9+**: The entire code is based on Python. It is recommended to use the latest version.

- **Batsim**: The main HPC simulation tool, installed following the official instructions from the [Batsim GitHub repository](https://github.com/oar-team/batsim).

- **Required Python libraries**, commonly including:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - pybatsim (or other custom Batsim interfaces)
    - evalys
    - sortedcontainers
    - xgboost
    - lightgbm
    - catboost
    - And other dependencies as needed
---

## **Installation:**

Follow these steps to set up your environment:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/baodangtrandev/HPC_simulation.git
    cd HPC_simulation
    ```

2. **Create and activate a virtual environment** (recommended). You can use either conda or venv:

    - *With conda:*
        ```bash
        conda activate <your_virtual_environment>
        ```

    - *With venv:*
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Linux/macOS
        venv\Scripts\activate     # On Windows
        ```

3. **Install required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install Batsim** following the official instructions [Batsim Installation](https://batsim.readthedocs.io/en/latest/installation.html). Make sure you can run the `batsim` command from your terminal.

---

## **Usage Guide:**

To run a complete simulation, follow these steps:

1. **Prepare the workload data:**  
   Place your workload files in the appropriate directory.

2. **Initialize and run the simulation with Batsim**  
   (For example, running the HCMUT-SuperNodeXP-2017 workload and the cluster512 platform):
    ```bash
    batsim -p ./simulation/platform/cluster512.xml \
           -w ./simulation/workload/user-estimate/HCMUT-SuperNodeXP-2017.json \
           --disable-schedule-tracing \
           --disable-machine-state-tracing
    ```

3. **Run the scheduling process using the HDEM prediction model and easy_backfilling:**
    ```bash
    pybatsim ./simulation/schedulers/easy_hdem.py -t 999999999
    ```

4. **Analyze results and perform visualization:**  
   Use your preferred tools or scripts in the repository to analyze and visualize the simulation outcomes.

---

## **Contribution & Contact:**

If you encounter any issues or would like to contribute, please open an issue or a pull request on this repository. All feedback and contributions are welcomed to improve the project.

---

## **References:**

- [Batsim Documentation](https://batsim.readthedocs.io/)
- HDEM Model Paper/Implementation (Comming Soon)
- [Backfilling Scheduling Algorithm](https://en.wikipedia.org/wiki/Backfilling_(scheduling))

---

**We hope you have a productive experience simulating HPC environments with this project!**
