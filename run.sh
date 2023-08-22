#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=01:00
#SBATCH --account=eecs471w23_class
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

# The application(s) to execute along with its input arguments and options:

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: ./run_tests {program}"
    echo "Usage: ./run_tests {program} {test_number}"
    exit 1
fi

export LD_LIBRARY_PATH=$PWD/libwb/build/:$LD_LIBRARY_PATH

if [ "$#" -eq  1 ]; then
    SCORE=0
    TOTAL=0
    for test in test/*; do
        ((++TOTAL))
        if diff <(./$1 -e $test/output.ppm -i $test/input.ppm -o $test/attempt.ppm -t image | tail -1) <(echo The solution is correct) &>/dev/null; 
        then
            ((++SCORE))
            echo "Passed $test!"
        else
            echo "$test is incorrect..."
        fi
    done

    echo "$SCORE/$TOTAL tests passed"
fi

if [ "$#" -eq  2 ]; then
    ./$1 -e test/$2/output.ppm -i test/$2/input.ppm -o test/$2/attempt.ppm -t image
fi