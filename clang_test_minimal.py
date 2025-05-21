import os
import clang.cindex

# Set up clang configuration
if os.path.exists(r"C:\Program Files\LLVM\bin\libclang.dll"):
    print("Found libclang.dll")
    clang.cindex.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")
else:
    print("Could not find libclang.dll")

# Try to create an index
try:
    index = clang.cindex.Index.create()
    print("Successfully created clang Index")
except Exception as e:
    print(f"Error creating Index: {e}")
