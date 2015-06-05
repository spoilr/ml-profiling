var svg = d3.select(".legend").append("svg")
                    .attr("width", 150)
                    .attr("height", 100);

 
var circle = svg.append("circle")
          .attr("cx", 30)
          .attr("cy", 30)
          .attr("r", 10)
          .attr("fill", "#DC9596");
          // .text("evidence");

var circle = svg.append("circle")
          .attr("cx", 30)
          .attr("cy", 60)
          .attr("r", 10)
          .attr("fill", "#F0FFCE");
          // .text("above threshold");

var circle = svg.append("circle")
          .attr("cx", 30)
          .attr("cy", 90)
          .attr("r", 10)
          .attr("fill", "#197BBD");
          // .text("below threshold");  


svg.selectAll("text")
    .data([0,1,2])
    .enter()
    .append("text")
    .attr("x", function(d) { return 50; })
    .attr("y", function(d) { 
    	if (d===0) {
    		return 34;
    	} else if (d===1) {
    		return 64;
    	} else {
    		return 95;
    	}
    })
    .text(function(d){
    	if (d===0) {
    		return "evidence";
    	} else if (d===1) {
    		return "above threshold";
    	} else {
    		return "below threshold";
    	}
    });