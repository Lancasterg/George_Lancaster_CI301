(function() {
    var canvas = document.querySelector("#canvas");
    var context = canvas.getContext("2d");
    //Canvas Dimensions
    var width = screen.width * 0.34;
    var height = screen.height * 0.7;
    canvas.width = width;
    canvas.height = height;
    MAX_HEIGHT = height;
    MAX_WIDTH = width;

    var Mouse = {
        x: 0,
        y: 0
    };
    var lastMouse = {
        x: 0,
        y: 0
    };
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.color = "black";
    context.lineWidth = 3;
    context.lineJoin = context.lineCap = "round";

    setup();

    canvas.addEventListener("mousemove", function(e) {
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;

        Mouse.x = e.pageX - this.offsetLeft;
        Mouse.y = e.pageY - this.offsetTop;

    }, false);

    canvas.addEventListener("mousedown", function(e) {
        canvas.addEventListener("mousemove", onPaint, false);
    }, false);

    canvas.addEventListener("mouseup", function() {
        canvas.removeEventListener("mousemove", onPaint, false);
    }, false);

		// Draw where mouse is
    var onPaint = function() {
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo(lastMouse.x, lastMouse.y);
        context.lineTo(Mouse.x, Mouse.y);
        context.closePath();
        context.stroke();
    };


    // Set up clear and load
    function setup() {
        // reset the clear button on click
        $("#clearButton").click(function() {
            context.clearRect(0, 0, 280, 280);
            canvas.width = MAX_WIDTH;
            canvas.height = MAX_HEIGHT;
            context.fillStyle = "white";
            context.fillRect(0, 0, canvas.width, canvas.height);
            context.lineWidth = 3;
        });

        // Upload and image
        var imageLoader = document.getElementById("imageLoader");
        imageLoader.addEventListener("change", handleImage, false);

        function handleImage(e) {
            var reader = new FileReader();
            reader.onload = function(event) {
                var img = new Image();

                img.onload = function() {

                    if (img.width > img.height) {
                        if (img.width > MAX_WIDTH) {
                            img.height *= MAX_WIDTH / img.width;
                            img.width = MAX_WIDTH;
                        }
                    } else {
                        if (img.height > MAX_HEIGHT) {
                            img.width *= MAX_HEIGHT / img.height;
                            img.height = MAX_HEIGHT;
                        }
                    }
                    context.clearRect(0, 0, canvas.width, canvas.height);
                    canvas.width = img.width;
                    canvas.height = img.height;
                    context.drawImage(img, 0, 0, img.width, img.height);
                }
                img.src = event.target.result;
            }
            reader.readAsDataURL(e.target.files[0]);
            context.lineWidth = 3;
        }
    }
}());
