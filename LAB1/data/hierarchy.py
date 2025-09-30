import os
import argparse

def print_directory_tree(start_path='.', file_limit=3, ignored_dirs=None):
    """
    Prints the directory tree of a given path in ASCII format,
    focusing on folders, but showing a limited list of files.

    :param start_path: The path to start the traversal from (default is current directory).
    :param file_limit: Maximum number of files to show (if exceeded, '...' is added).
    :param ignored_dirs: A list of directory names to ignore during traversal.
    """
    if ignored_dirs is None:
        ignored_dirs = []

    default_ignored = ['.git', '__pycache__', '.ipynb_checkpoints', 'node_modules', '.idea']
    ignored_dirs = list(set(ignored_dirs + default_ignored))

    print(f"** Directory Tree for: {os.path.abspath(start_path)} **\n")

    def tree(root, prefix=''):
        try:
            contents = os.listdir(root)
        except OSError as e:
            print(f"{prefix}└── [ERROR: Cannot access {root} - {e}]")
            return

        dirs = sorted([d for d in contents if os.path.isdir(os.path.join(root, d)) and d not in ignored_dirs])
        files = sorted([f for f in contents if os.path.isfile(os.path.join(root, f))])

        files_to_display = files[:file_limit]
        show_ellipsis = (len(files) > file_limit)

        entries = dirs + files_to_display
        if show_ellipsis:
            entries.append("...")

        n_entries = len(entries)

        for i, entry in enumerate(entries):
            is_last = (i == n_entries - 1)
            connector = '└── ' if is_last else '├── '
            print(f"{prefix}{connector}{entry}")

            if entry in dirs:
                full_path = os.path.join(root, entry)
                next_prefix = prefix + ('    ' if is_last else '│   ')
                tree(full_path, next_prefix)

    tree(start_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print a directory tree with file limits.")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="The directory path to start from (default: current directory)"
    )
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=3,
        help="Maximum number of files to show per directory (default: 3)"
    )
    parser.add_argument(
        "-i", "--ignore",
        nargs="*",
        default=[],
        help="List of additional directories to ignore"
    )
    
    args = parser.parse_args()
    print_directory_tree(start_path=args.path, file_limit=args.num_files, ignored_dirs=args.ignore)
