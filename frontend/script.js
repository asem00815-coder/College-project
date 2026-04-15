new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Экономика регионов', 
            data: [9, 7, 9],
            backgroundColor: ['red', 'blue', 'green']
        }]
    },
options: {
    plugins: {
        title: {
            display: true,
            text: 'Экономика регионов' 
        }
    }
}
});

new Chart(document.getElementById('pieChart'), {
    type: 'pie',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Торговля регионов в процентах',
            data: [20, 50, 30],
            backgroundColor: ['red', 'blue', 'green']
        }]
},
options: {
    plugins: {
        title: {
            display: true,
            text: 'Торговля регионов' 
        }
    }
}
});