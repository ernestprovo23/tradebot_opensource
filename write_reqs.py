import subprocess

# Run pip freeze and capture the output
output = subprocess.check_output(['pip', 'freeze'])

# Decode the output from bytes to string
decoded_output = output.decode('utf-8')

# Split the output into lines and remove version numbers
packages = [line.split('==')[0] for line in decoded_output.splitlines()]

# Write the package names to requirements.txt
with open('requirements.txt', 'w') as f:
    for package in packages:
        f.write(f"{package}\n")

print("Packages have been written to requirements.txt without version numbers.")
