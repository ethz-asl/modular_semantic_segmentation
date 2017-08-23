#! /bin/bash
basedir=$(grep -Po "DATA_BASEPATH = '\K[^']*" "./xview/settings.py")

if [ ! -d "$basedir" ]; then
echo "ERROR: Specified directory path $basedir does not exist. Please try again after creating the directory."
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
# End of perl script
# Now we extract the weights from the archive and remove the archive.
unzip "$weightsfile" -d "$basedir/darnn" > /dev/null
rm "$weightsfile"
echo "INFO: DA-RNN weights downloaded into $basedir/darnn."
fi

if [ ! -d "$basedir/synthia" ]; then
mkdir "$basedir/synthia"
fi
echo "INFO: Checking for exisiting SYNTHIA data"

# This is the list of all available SYNTHIA data sequences.
declare -a sequences=(
    "SYNTHIA-SEQS-04-DAWN"
    "SYNTHIA-SEQS-04-FALL"
    "SYNTHIA-SEQS-04-FOG"
    "SYNTHIA-SEQS-04-NIGHT"
    "SYNTHIA-SEQS-04-RAINNIGHT"
    "SYNTHIA-SEQS-04-SOFTRAIN"
    "SYNTHIA-SEQS-04-SPRING"
    "SYNTHIA-SEQS-04-SUMMER"
    "SYNTHIA-SEQS-04-SUNSET"
    "SYNTHIA-SEQS-04-WINTER"
    "SYNTHIA-SEQS-04-WINTERNIGHT")
declare -a downloadkeys=(
    "6a9d6eaa25cc5bf7f3bb1a15ecad8757"
    "242d33b6711b0c624284f030c494dbae"
    "7924cca33c2c658504af2b5031a83e2e"
    "07db026fa9ad2c270febb8cb13cd3855"
    "c8bd2e69b2bc34ea9013eacb2bb86083"
    "309ae57a9cb76146ecfcaef6acfc00ca"
    "e546e22d653130eeef8143bb8a6901a2"
    "1082fe786a09e2fe425c323a50913d1e"
    "dffc1ee818ef73fb37f61d3bcf95ef83"
    "43ef9a1ba04bdce9ed1fe4cb5b76b7e0"
    "82472575e7d12de4a97238d5a0b13fec")
## Now we look through the list and download all sets that are not yet downloaded.
for i in "${!sequences[@]}"
do
    if [ -d "$basedir/synthia/${sequences[$i]}" ]; then
        echo "INFO: ${sequences[$i]} found, wont download again."
        echo "      If there are problems please remove all files from $basedir/synthia/${sequences[$i]} and start this script again."
        continue;
    fi
    echo "INFO: ${sequences[$i]} not found, start downloading..."

    wget -O "$basedir/synthia/data.rar" "http://synthia-dataset.net/wp-content/plugins/email-before-download/?dl=${downloadkeys[$i]}" 
    echo "INFO: Extracting files..."
    unrar x "$basedir/synthia/data.rar" "$basedir/synthia" > /dev/null
    # remove all unncessessary files
    declare -a unnecessary=(
        "Stereo_Left"
        "Stereo_Right/Omni_L"
        "Stereo_Right/Omni_R"
        "Stereo_Right/Omni_B")
    for d in "${unnecessary[@]}"
    do
        rm -rf "$basedir/synthia/${sequences[$i]}/RGB/$d"
        rm -rf "$basedir/synthia/${sequences[$i]}/Depth/$d"
        rm -rf "$basedir/synthia/${sequences[$i]}/GT/COLOR/$d" "$basedir/synthia/${sequences[$i]}/GT/LABELS/$d"
    done
    echo "INFO: ${sequences[$i]} successfully extracted"
done
# remove the downloaded archive file
rm "$basedir/synthia/data.rar"

echo "INFO: Data successfully downloaded."

exit 0