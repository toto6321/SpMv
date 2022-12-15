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
    formats you choose. If you programme on the MATLAB, then you will choose \
    MATLAB and go to the mat folder, while in our case we use the matrix \
    market extension so that we go to the MM folder.
'''

'''
2.  Execute this script with source data path and destination path.
'''


def get_dataset_ready(source_path=None, des_path=None):
    if os.path.isdir(source_path):
        for dirpath, dirnames, files in os.walk(source_path):
            for file in files:
                abspath = os.path.join(dirpath, file)

                pass
                # if it is a tarball file
                if file.endswith('.tar.gz'):
                    target_path = os.path.normpath(
                        os.path.join(
                            des_path,
                            os.path.relpath(dirpath, start=source)
                        )
                    )
                    print('target path:', target_path)

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
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tarobj, target_path)
                        print('successful to extract file ', file, ' to ',
                              target_path)


if __name__ == '__main__':
    '''
        :param source: parent directory containing matrices 
        :param destination: parent directory to save extracted files 
    '''

    root_path = os.getcwd()
    if len(sys.argv) > 2:
        source = sys.argv[1]
        destination = sys.argv[2]
    elif len(sys.argv) > 1:
        source = sys.argv[1]
        destination = root_path + '/dataset'
    else:
        source = root_path + '/test/source'
        destination = root_path + '/test/dataset'

    get_dataset_ready(source, destination)
