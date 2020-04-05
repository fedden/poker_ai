import Vue from 'vue'
import App from './App'
import router from './router'

Vue.config.productionTip = false

new Vue({
  el: '#app',
  router,
  render: function (createElement) {
    return createElement(App, {
      props: {
        player_playing: this.player_playing,
        players: this.players,
        figures: this.figures,
        values: this.values,
        cards: this.cards,
        five_cards: this.five_cards,
      },
    })
  },
  data: {
    player_playing: 0,
    players: [
      { name: 'rivy33', bank: 100, onTable: 77, hasCards: false },
      {
        name: 'kattar',
        color: 'cyan',
        bank: 400,
        onTable: 300,
        hasCards: true,
      },
      {
        name: 'mikelaire',
        color: 'lightcoral',
        bank: 100,
        onTable: 20,
        hasCards: false,
      },
      {
        name: 'tomtom',
        color: 'crimson',
        bank: 100,
        onTable: 20,
        hasCards: false,
      },
      { name: 'nana', color: '#444', bank: 100, onTable: 20, hasCards: false },
      // {name:'ionion', color: 'forestgreen', bank: 100, onTable: 20, hasCards: false},
      // {name:'link6996', color: 'goldenrod', bank: 100, onTable: 20, hasCards: false},
      // {name:'gossboganon', color: 'gold', bank: 100, onTable: 20, hasCards: false}
    ],
    figures: ['P', 'H', 'C', 'D'],
    values: ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'],
  },
  computed: {
    cards() {
      let all = []
      for (let figure of this.figures) {
        for (let value of this.values) {
          all.push({
            f: figure,
            v: value,
          })
        }
      }
      return all
    },
    five_cards() {
      let fives = []
      for (let i = 0; i < 5; i++) {
        let rand_id = parseInt(Math.random() * this.cards.length)
        fives.push(this.cards[rand_id])
      }
      return fives
    },
  },
})
