import os
import sys
import tarfile

'''
1.  First use the interface provided by the SuiteSparseMatrix Collection \
    https://sparse.tamu.edu/interfaces to download the the data/matrices \
    upon your appetite.
    
    For example, you could use the Java Graphical Interface (ssgui).
    When the download starts, it will create four folders in your home \
    directory by default. (unfortunately, there is no other choice.)
   
    They are files, MM, mat and boeing. Image files of the view of the \
    matrices are saved in the files folder, while the tarball files of the \
    matrices are saved in either MM, mat or boeing depending on which \
    formats you choose. If you programme on the Matlab, then you will choose \
    matlab and go to the mat folder, while in our case we use the matrix \
    market extension so that we go to the MM folder.
'''

'''
2.  Copy this script to the MM folder and execute it after you determine \
    the target path, which is the <project_path>/suite_sparse_dataset/.
    
    Alternatively, you can execute this script with the path of MM folder \
    assigned.
'''

root_path = os.getcwd()
dataset_folder = 'suite_matrix_dataset/MM'
dataset_abspath = os.path.normpath(os.path.join(root_path, dataset_folder))

# if the MM folder's path is given
if len(sys.argv) > 1:
    data_path = sys.argv[1]
    if os.path.isdir(data_path):
        for dirpath, dirnames, files in os.walk(data_path):
            for file in files:
                abspath = os.path.normpath(os.path.join(dirpath, file))

                # if it is a tarball file
                if file.endswith('.tar.gz'):
                    target_path = os.path.normpath(
                        os.path.join(
                            dataset_abspath,
                            os.path.relpath(dirpath, start=data_path)
                        )
                    )
                    print('target data path', target_path)

                    # create the folder tree
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)

                    file_name = file[0:file.find('.tar.gz')]
                    target_folder = os.path.join(target_path, file_name)

                    # end this iteration if the file has been extracted
                    if os.path.exists(target_folder):
                        print(file, ' has been extracted! Skip!')
                        continue

                    # else extract the file
                    with tarfile.open(abspath) as tarobj:
                        tarobj.extractall(target_path)
                        print('successful to extract file ', file, ' to ',
                              target_path)
else:
    print('no path of the MM folder is provided.')
