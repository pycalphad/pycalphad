"""
This module print out static HTML elements for hardcoded installation instructions
"""
package_managers = ['pip', 'conda', 'source']
platform = ['linux', 'windows']
env_flag = ['noenv', 'env']
jupyter_flag = ['nojupyter', 'jupyter']
import itertools

ENVIRONMENT_NAME = 'pycalphad-env'

for config_tuple in itertools.product(package_managers, platform, env_flag, jupyter_flag):
    pkg, plt, env, jup = config_tuple
    name = '_'.join(config_tuple)

    install_lines = []
    prefmt_string = ''

    if env == 'env':
        # Create environment
        if pkg == 'conda':
            install_lines.append(f"conda create -n {ENVIRONMENT_NAME} python")
        else:
            install_lines.append(f"python -m venv {ENVIRONMENT_NAME}")

        # Activate environment
        if pkg == 'conda':
            install_lines.append(f"conda activate {ENVIRONMENT_NAME}")
        else:
            if plt == 'windows':
                install_lines.append(f"{ENVIRONMENT_NAME}\\Scripts\\activate")
            else:
                install_lines.append(f"source {ENVIRONMENT_NAME}/bin/activate")

    # Install pycalphad with optional JupyterLab
    # Should not end with newline
    install_string = ""
    if pkg == 'conda':
        install_line = "conda install pycalphad"
        if jup == "jupyter":
            install_line += " jupyterlab"
        install_lines.append(install_line)
    elif pkg == 'source':
        install_lines.append("git clone https://github.com/pycalphad/pycalphad")
        install_lines.append("cd pycalphad")
        install_lines.append("pip install -e .")  # TODO: setup.py build_ext --inplace ?
        if jup == "jupyter":
            install_lines.append("pip install jupyterlab")
    else:  # assume pip
        install_line = "pip install pycalphad"
        if jup == "jupyter":
            install_line += " jupyterlab"
        install_lines.append(install_line)

    # prepend "$ " to lines
    install_lines = ["$ " + line for line in install_lines]

    preformatted_string = "\n".join(install_lines)
    s = f'<pre id={name} hidden>{preformatted_string}</pre>'
    print(s)
    print()
#    print('_'.join(config_tuple))
