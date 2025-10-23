"""
This module print out static HTML elements for hardcoded installation instructions
"""

import itertools
from typing import List

# Hardcoded content
ENVIRONMENT_NAME = 'pycalphad-env'
# For integration into external tools, like Sphinx:
HTML_TEMPLATE = """<div id={} hidden><div class="highlight-bash notranslate"><div class="highlight"><pre>{}</pre></div></div></div>"""
ENTRIES_JOIN_STR = ""  # join between different HTML entries

# Matrix of options
PACKAGE_MANAGERS = ['pip', 'conda', 'source']
PLATFORMS = ['linux', 'windows']
ENV_FLAGS = ['noenv', 'env']
JUPYTER_FLAGS = ['nojupyter', 'jupyter']


def get_config_install_lines(pkg, plt, env, jup) -> List[str]:
    """Take the configuration and produce a list of lines to install that configuration"""
    install_lines = []

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
    if pkg == 'conda':
        install_line = "conda install -c conda-forge pycalphad"
        if jup == "jupyter":
            install_line += " jupyterlab"
        install_lines.append(install_line)
    elif pkg == 'source':
        install_lines.append("git clone https://github.com/pycalphad/pycalphad")
        install_lines.append("cd pycalphad")
        install_lines.append("pip install -U pip setuptools")
        install_lines.append("pip install -U -r requirements-dev.txt")
        install_lines.append("pip install -U --no-build-isolation --editable .")
        if jup == "jupyter":
            install_lines.append("pip install -U jupyterlab")
    else:  # assume pip
        install_line = "pip install -U pycalphad"
        if jup == "jupyter":
            install_line += " jupyterlab"
        install_lines.append(install_line)
    return install_lines


def get_config_html(package_manager, platform, env_flag, jupyter_flag, prepend_shell_prompt=False):
    """Get HTML for a single configuration"""
    install_lines = get_config_install_lines(package_manager, platform, env_flag, jupyter_flag)

    if prepend_shell_prompt:
        if platform == 'windows':
            install_lines = ["> " + line for line in install_lines]
        else:
            install_lines = ["$ " + line for line in install_lines]

    # Build the HTML
    config_id = '_'.join((package_manager, platform, env_flag, jupyter_flag))
    formatted_string = "<br>".join(install_lines)
    generated_config_html = HTML_TEMPLATE.format(config_id, formatted_string)
    return generated_config_html

def get_matrix_html(package_managers, platform, env_flags, jupyter_flags, wrap=True):
    html_entries = []
    for config_tuple in itertools.product(package_managers, platform, env_flags, jupyter_flags):
        html_entries.append(get_config_html(*config_tuple))
    if wrap:
        html_entries.insert(0, '<div id="install-samples">')
        html_entries.append('</div>')
    return ENTRIES_JOIN_STR.join(html_entries)

if __name__ == '__main__':
    print(get_matrix_html(PACKAGE_MANAGERS, PLATFORMS, ENV_FLAGS, JUPYTER_FLAGS))
