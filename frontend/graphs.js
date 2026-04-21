new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{ 
            label: 'Насыщенность внутреннего рынка',
            data: [9, 7, 9],
            borderColor: 'white',
            backgroundColor: ['red', 'blue', 'green']
        }]
    },
options: {
    scales: {
        y: {
            beginAtZero: true,
            max: 10
        }
    },
     plugins: {
        title: {
            display: true,
            text: '',
        }
    }
}
});

new Chart(document.getElementById('pieChart'), {
    type: 'pie',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            data: [2, 5, 3],
            borderColor: 'white',
            backgroundColor: ['red', 'blue', 'green']
        }]
},
options: {
    plugins: {
        title: {
            display: true,
            text: 'Доля во внешней торговле' 
        }
    }
}
});
new Chart(document.getElementById('bar2Chart'), {
    type: 'bar',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Степень ограничения власти князя', 
            data: [3, 10, 5],
            borderColor: 'white',
            backgroundColor: ['red', 'blue', 'green']
        }]
    },
options: {
    plugins: {
        title: {
            display: true,
        }
    }
}
});
new Chart(document.getElementById('pie2Chart'), {
    type: 'pie',
    data: {
        labels: ['Ростово-суздальское княжество', 'Новгородское княжество', 'Галицко-Волынское княжество'],
        datasets: [{
            label: 'Влияние на общерусскую политику',
            data: [9, 5, 9],
            borderColor: 'white',
            backgroundColor: ['red', 'blue', 'green']
        }]
},
options: {
    plugins: {
        title: {
            display: true,
            text: 'Влияние на общерусскую политику'
        }
    }
}});