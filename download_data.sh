#! /bin/bash
basedir=$1

if [ ! -d "$basedir" ]; then
echo "ERROR: Specified directory does not exist. Please try again after creating the directory"
exit 0
fi

if [ ! -d "$basedir/darnn" ]; then
echo "INFO: Downloading DA-RNN weights"
mkdir "$basedir/darnn"

# To download the large weights file from google-drive, we require the following perl script.
weightsfile="$basedir/darnn/weights.zip"
perl - $weightsfile << '__HERE__'

use strict;

my $COMMAND;
my $confirm;
my $check;
sub execute_command();

my $FILENAME = shift;
my $URL="https://drive.google.com/uc?export=download&id=0B4WdmTHU8V7VQWFnRmFIVTA1LXc";

execute_command();

while (-s $FILENAME < 100000) { # only if the file isn't the download yet
    open fFILENAME, '<', $FILENAME;
    $check=0;
    foreach (<fFILENAME>) {
        if (/href="(\/uc\?export=download[^"]+)/) {
            $URL='https://docs.google.com'.$1;
            $URL=~s/&amp;/&/g;
            $confirm='';
            $check=1;
            last;
        }
        if (/confirm=([^;&]+)/) {
            $confirm=$1;
            $check=1;
            last;
        }
        if (/"downloadUrl":"([^"]+)/) {
            $URL=$1;
            $URL=~s/\\u003d/=/g;
            $URL=~s/\\u0026/&/g;
            $confirm='';
            $check=1;
            last;
        }
    }
    close fFILENAME;
    die "Couldn't download the file :-(\n" if ($check==0);
    $URL=~s/confirm=([^;&]+)/confirm=$confirm/ if $confirm ne '';

    execute_command();
}

sub execute_command() {
    $COMMAND="wget -nv --load-cookie /tmp/cookie.txt --save-cookie /tmp/cookie.txt \"$URL\"";
    $COMMAND.=" -O \"$FILENAME\"" if $FILENAME ne '';
    `$COMMAND`;
    return 1;
}
__HERE__
unzip "$weightsfile" -d "$basedir/darnn" > /dev/null
rm "$weightsfile"
echo "INFO: DA-RNN weights downloaded into $basedir/darnn."
fi

rm ./xview/__init__.py
echo "DATA_BASEPATH = '$basedir' " > ./xview/__init__.py
echo "INFO: Data successfully downloaded and linked."

exit 0