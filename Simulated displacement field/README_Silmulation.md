### README for Extracting Surface Node Displacement from Abaqus ODB Files

This Python script is designed to extract the displacement data for surface nodes from an Abaqus `.odb` (output database) file and write it into a CSV file. The script specifically extracts the X and Y displacement components of nodes from a defined node set.

#### Requirements

- **Abaqus/CAE** with Python scripting environment (`odbAccess` module available).
- **Python**(typically, the version bundled with Abaqus for scripting).
- **Abaqus `.odb` file**: The script works with Abaqus output databases (ODB files).
- **CSV support**: The output of the script is written to a CSV file.

#### Files Needed

- **ODB File**: The Abaqus ODB file from which the surface node displacements are to be extracted. In the example provided, the file is: `Job.odb`.
- **generation_demo_dis_x_y.py**:Extract displacement field from Abaqus.odb file.
- **generation_abaqus_field.py**:Displacement data is written to a .csv file.In the example provided, the file is: `demo_dis_x_y.csv`.



