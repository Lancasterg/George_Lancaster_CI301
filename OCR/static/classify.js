// Get the modal
var modal = document.getElementById('myModal');

// Get the button that opens the modal
var btn = document.getElementById("myBtn");
var cropImg = document.getElementById("cropImage");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];
var canvas = document.getElementById("canvas");
var cropper;

// Begin cropping
function beginCrop() {
    modal.style.display = "block";
    canvas = document.getElementById("canvas");
    cropImg.src = canvas.toDataURL();
    imgWidth = cropImg.width;
    imgHeight = cropImg.height;

    cropper = new Cropper(cropImg, {
        movable: false,
        zoomable: false,
        rotatable: false,
        scalable: false
    });
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
    cropper.destroy();
    modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
    if (event.target == modal) {
        cropper.destroy();
        modal.style.display = "none";

    }
}
// Crop the area specified
function getCroppedArea() {
    var img = cropper.getCroppedCanvas();
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0, img.width, img.height);
    cropper.destroy();
    modal.style.display = "none";
}

// Start a download for the client using the result of last classification
function downloadTheResult() {
    var filename = "result.txt";
    var element = document.createElement('a');
    var data = document.getElementById("result").value;
    if (data != "") {
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + data);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    } else {
        window.alert("Classify an image first!");
    }
}
// Set the type of classification
function setNetwork() {

    var networkType = document.getElementById("networkSelect").value
    if (networkType != "none") {
        $.ajax({
            type: "POST",
            url: "/setNetwork/",
            data: networkType,
            success: function(response) {
                console.log(response);
            }
        });
    }
}
// Make a prediction
function getPrediction() {

    var canvasObj = document.getElementById("canvas");
    var img = canvasObj.toDataURL();
    var networkType = document.getElementById("networkSelect").value;
    $.ajax({
        type: "POST",
        url: "/predict/",
        data: img,
        success: function(data) {
            $('#result').text(data);
            console.log(screen.width * 0.4);
            document.getElementById("loader").style.display = "none";
        }
    });
}
// Make a predciton
function predict() {
    var networkType = document.getElementById("networkSelect").value;
    if (networkType != "none") {
        document.getElementById("loader").style.display = "block";
        setNetwork();
        getPrediction();
    } else {
        window.alert("Select a network type first.");
    }
}
// Save an image
function saveTheImage() {
    var canvasObj = document.getElementById("canvas");
    var img = canvasObj.toDataURL();
    $.ajax({
        type: "POST",
        url: "/save_result/",
        data: img,
        success: function() {
            console.log("success");
        }
    });
}

$("#modalCrop").click(getCroppedArea);
$("#myButton").click(predict);
$("#cropButton").click(beginCrop);
$("#downloadButton").click(downloadTheResult);
