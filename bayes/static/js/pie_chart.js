var create_data = function(inference, feature) {
  if (feature === null) {
    feature = 'HighValueCivilian'
  }

  feature = feature.replace(/\s/g,'');  
  var probs = inference[feature]

  var likelihood = []
  for (var key in probs) {
    var feature_prob = []
    feature_prob.push(key)
    feature_prob.push(probs[key])
    likelihood.push(feature_prob)
  }

  data = [{
    type: 'pie',
    data: likelihood
  }]
  return data
}

var pie_chart = function(inference, feature) {

  (function($) {
    $(function () {

        if (feature === null) {
          feature = 'HighValueCivilian'
        }
        data = create_data(inference, feature)        

        $('#pie').highcharts({
          chart: {
            plotBackgroundColor: null,
            plotBorderWidth: null,
            plotShadow: false
          },
          title: {
            text: 'Distribution of feature ' + feature 
          },
          tooltip: {
              pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
          },
          plotOptions: {
            pie: {
                allowPointSelect: true,
                cursor: 'pointer',
                dataLabels: {
                    enabled: true,
                    format: '<b>{point.name}</b>: {point.percentage:.1f} %',
                    style: {
                        color: (Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black'
                    }
                }
            }
          },
          series: data
        });


        $('.distribution').on('click',function(e) {
            e.preventDefault();
            $( "#pie" ).empty();
            feature = $(this).parent().siblings().eq(0).text();
            // pie_chart(inference, feature);
            data = create_data(inference, feature)
            $('#pie').highcharts({
              chart: {
                plotBackgroundColor: null,
                plotBorderWidth: null,
                plotShadow: false
              },
              title: {
                text: 'Distribution of feature ' + feature 
              },
              tooltip: {
                  pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
              },
              plotOptions: {
                pie: {
                    allowPointSelect: true,
                    cursor: 'pointer',
                    dataLabels: {
                        enabled: true,
                        format: '<b>{point.name}</b>: {point.percentage:.1f} %',
                        style: {
                            color: (Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black'
                        }
                    }
                }
              },
              series: data
            });
          });

    });
  })(jQuery);

}
