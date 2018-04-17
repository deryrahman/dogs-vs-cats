$(document).ready(function(){
  $("#uploader").on('submit', function(e){
    e.preventDefault();
    var form = $('form')[0];
    var formData = new FormData(form);
    $.ajax({
      url: 'predict',
      type : "POST",
      contentType: false,
      processData: false,
      data : formData,
      success : function(result) {
        console.log(result);
        $('#predicted').empty()
        $('#predicted-rival').empty()
        cat = result['payload']['predicted']['cats']
        dog = result['payload']['predicted']['dogs']
        predicted = 'Dog'
        predictedRival = 'Cat'
        percentage = dog * 100
        percentageRival = cat * 100
        if(cat > dog){
          percentage = cat * 100
          percentageRival = dog * 100
          predicted = 'Cat'
          predictedRival = 'Dog'
        }
        $('img').attr('src', result['payload']['path'])
        $('img').addClass('img-thumbnail')
        $('#predicted').append(predicted + " : " + percentage + "%")
        $('#predicted-rival').append(predictedRival + " : " + percentageRival + "%")
      },
      error: function(xhr, resp, text) {
          console.log(xhr, resp, text);
      }
    })
  });
});
