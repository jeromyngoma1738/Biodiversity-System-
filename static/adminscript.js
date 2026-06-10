// sidebar toggle
var sidebarOpen = false;
var sidebar = document.getElementById("sidebar");

function openSidebar() {
    if (!sidebarOpen) {
        sidebar.classList.add("sidebar-responsive"); // fixed spelling
        sidebarOpen = true;
    }
}

function closeSidebar() {
    if (sidebarOpen) {
        sidebar.classList.remove("sidebar-responsive"); // fixed spacing
        sidebarOpen = false;
    }
}


//CODE FOR  BAR CHARTS

var barChartOption = {
   series: [{
    data: [400, 430, 448, 470, 540,]
  }],
  chart: {
    type: 'bar',
    height: 400
  },
  plotOptions: {
    bar: {
      borderRadius: 4,
      borderRadiusApplication: 'end',
      horizontal: true,
    }
  },
  dataLabels: {
    enabled: false
   },
   xaxis: {
    categories: [ 'animals', 'cats', 'animals', 'animals',

     ],
     }
     };
     var chart = new ApexCharts(document.querySelector("#bar-chart"), barChartOption);
     chart.render();
  
// area chart
// AREA CHART DATA
var salesData = [31, 40, 28, 51, 42, 109, 100];
var months = [
  "2024-01-01",
  "2024-02-01",
  "2024-03-01",
  "2024-04-01",
  "2024-05-01",
  "2024-06-01",
  "2024-07-01"
];

// AREA CHART OPTIONS
var areaChartOptions = {
  series: [{
    name: "Sales",
    data: salesData
  }],

  chart: {
    type: 'area',
    height: 350,
    toolbar: { show: false }
  },

  colors: ["#00ab57"],

  dataLabels: { enabled: false },

  stroke: {
    curve: 'smooth',
    width: 3
  },

  fill: {
    type: "gradient",
    gradient: {
      shadeIntensity: 1,
      opacityFrom: 0.4,
      opacityTo: 0.1
    }
  },

  title: {
    text: "Trend Observation",
    align: "left"
  },

  grid: {
    borderColor: "#f4f4f4",
    strokeDashArray: 4
  },

  xaxis: {
    type: 'ob',
    categories: months
  },

  tooltip: {
    x: {
      format: "Species"
    }
  }
};

var areaChart = new ApexCharts(
  document.querySelector("#area-chart"),
  areaChartOptions
);
areaChart.render();