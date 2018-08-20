// Download the entry
function setDownload(pred_id){
  var send = pred_id.toString();
  $.ajax({
    type: "POST",
    url: "/download_result/",
    data: send,
    success: function(data){
        console.log(data);
        var filename = "result.txt"
        var element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + data);
        element.setAttribute('download', filename);
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    }
  });
}
// Delete the prediction
function deleteEntry(pred_id){
  var send = pred_id.toString();
  $.ajax({
    type: "POST",
    url: "/delete_result/",
    data: send,
    success: function(data){
      console.log(data)
      location.reload();
    }
  });
}
