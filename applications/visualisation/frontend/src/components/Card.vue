<template>
    <div class="card" :class="['figures-' + card.f, 'values-' + card.v]">
        <h1>{{card.v}}</h1>
        <div class="figures" :class="card.f"></div>
        <h1>{{card.v}}</h1>
    </div>
</template>


<script>
export default {
  name: 'Card',
  props: [
    'card'
  ]
}

</script>
