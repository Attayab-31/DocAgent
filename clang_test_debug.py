import os
import sys
import clang.cindex

print("Python path:", sys.path)
print("\nClang module:", clang.__file__)
print("\nChecking for libclang.dll...")

dll_path = r"C:\Program Files\LLVM\bin\libclang.dll"
if os.path.exists(dll_path):
    print(f"Found libclang.dll at: {dll_path}")
    try:
        clang.cindex.Config.set_library_file(dll_path)
        print("Successfully set library file")
    except Exception as e:
        print(f"Error setting library file: {e}")
else:
    print(f"libclang.dll not found at: {dll_path}")

print("\nTrying to create Index...")
try:
    index = clang.cindex.Index.create()
    print("Successfully created Index")
except Exception as e:
    print(f"Error creating Index: {e}")
    import traceback
    traceback.print_exc()
