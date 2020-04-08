<template>
    <div class="card-place">
        <Card v-for="(card, index) in five_cards" :key="index" :card="card"></Card>
    </div>
</template>


<script>
import Card from './Card.vue'

export default {
  name: 'CommunityCards',
  props: [
    'five_cards'
  ],
  components: {
    Card
  }
}

</script>
