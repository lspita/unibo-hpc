/****************************************************************************
 *
 * omp-brute-force.c - Brute-force password cracking
 *
 * Copyright (C) 2017--2022, 2024, 2025 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% Brute-force password cracking
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

![[DES cracker board](https://en.wikipedia.org/wiki/EFF_DES_cracker) developed
in 1998 by the Electronic Frontier Foundation (EFF); this device can be used to
brute-force a DES key. The original uploader was Matt Crypto at English
Wikipedia. Later versions were uploaded by Ed g2s at en.wikipedia - CC BY 3.0
us, <https://commons.wikimedia.org/w/index.php?curid=2437815>.](des-cracker.jpg)

The program [omp-brute-force.c](omp-brute-force.c) contains an
encrypted message stored in the array `enc[]` of length 64.  The
message has been encrypted using the _XOR_ algorithm, which applies
the "exclusive or" (xor) operator between the plaintext and the
encryption key. The _XOR_ algorithm is not secure, unless the key is
truly random and has the same length of the plaintext; however, it is
"good enough" for this exercise.

_XOR_ is a _symmetric_ encryption algorithm, meaning that the same key
is used for encryption and decryption. The function `xorcrypt(in, out,
n, key, keylen)` encrypts or decrypts a message with a given key. To
encrypt, `in` pointw to the plaintext and `out` to a memory buffer
where the ciphertext will be stored. To decrypt, `in` points to the
ciphertext and `out` to a memory buffer where the plaintext will be
stored.

The parameters are as follows:

- `in` points to the source message. This buffer does not need to be
  zero-terminated since it may contain an encrypted message.

- `out` points to a memory buffer of at least $n$ Bytes (the same
  length of the source message), that must be allocated by the caller.
  At the end, this buffer contains the source message that has been
  encrypted/decrypted with the encryption key.

- `n` is the length, in Bytes, of the source message.

- `key` points to the encryption/decryption key. The key is a sequence
  of arbitrary bytes, and therefore does not need to be
  zero-terminated.

- `keylen` is the length of the encryption/decryption key.

The _XOR_ algorithm will happily decrypt any message with any key; if
the key is not correct, the decrypted message will not make any
sense.

For this exercise we know that the plaintext is a zero-terminated
ASCII string that begins with `0123456789`. We also know that the
encryption key is a sequence of 8 ASCII numeric characters; therefore,
the key is a string from `"00000000"` and `"99999999"`. The goal is to
write a program to brute-force the key using OpenMP. The program
should try every key until a valid message is found, i.e., a message
that begins with `0123456789`. At the end, the program must print the
plaintext, which is a famous quote from [an old
film](https://en.wikipedia.org/wiki/WarGames).

The main loop can not be parallelized with the `omp for` construct
(why?). Therefore, we use `omp parallel` and manually partition the
key space among threads. Remember that `omp parallel` can be applied
to a _structured block_, i.e., a block with a single entry and a
single exit point. Therefore, the thread who finds the correct key can
not exit the block using `return`, `break` or `goto` (the compiler
should give a compile-time error). Howver, we need to terminate the
program at most shortly after the correct key is found. How?

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-brute-force.c -o
omp-brute-force

Run with:

        OMP_NUM_THREADS=2 ./omp-brute-force

**Note**: the execution time of the parallel program might change
irregularly depending on $P$. Why?

## Files

- [omp-brute-force.c](omp-brute-force.c)

***/

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Decrypt `enc` of length `n` bytes into buffer `dec` using `key` of
   length `keylen`. The encrypted message, decrypted messages and key
   are treated as binary blobs; hence, they do not need to be
   zero-terminated.

   Do not modify this function. */
void xorcrypt(const char* in, char* out, int n, const char* key, int keylen) {
  for (int i = 0; i < n; i++) {
    out[i] = (char)(in[i] ^ key[i % keylen]);  // xor outputs an int
  }
}

int main(void) {
  const int KEY_LEN = 8;
  /* encrypted message */
  const char enc[] = {4,   1,  0,  1,  0,  1,  4,  1,  12, 9,  115, 18, 71,
                      64,  64, 87, 90, 87, 87, 18, 83, 85, 95, 83,  26, 16,
                      102, 90, 81, 20, 93, 88, 88, 73, 18, 69, 93,  90, 92,
                      95,  90, 87, 18, 95, 91, 66, 87, 22, 93, 67,  18, 92,
                      91,  64, 18, 66, 91, 16, 66, 94, 85, 77, 28,  54};
  const int msglen = sizeof(enc);
  const char check[] = "0123456789"; /* the correctly decrypted message starts
                                        with these characters */
  const int CHECK_LEN = strlen(check);
  const int n = 100000000; /* number of possible keys */
  volatile int found = 0;
  char* out = (char*)malloc(msglen); /* where to put the decrypted message */
  assert(out != NULL);

  const double tstart = omp_get_wtime();
#pragma omp parallel default(none) \
    shared(KEY_LEN, msglen, n, found, enc, check, CHECK_LEN)
  {
    char key[KEY_LEN + 1]; /* sprintf will output the trailing \0, so we need
                             one byte more for the key */
    char* out = (char*)malloc(msglen);
    assert(out != NULL);

    const int my_id = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();
    const int my_start = (n * my_id) / num_threads;
    const int my_end = (n * (my_id + 1)) / num_threads;

    for (int k = my_start; k < my_end && !found; k++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
      snprintf(key, KEY_LEN + 1, "%08d", k);
#pragma GCC diagnostic pop
      xorcrypt(enc, out, msglen, key, KEY_LEN);
      /* `out` contains the decrypted text; if the key is not
         correct, `out` will contain garbage */
      if (0 == memcmp(out, check, CHECK_LEN)) {
        printf("Key found: %s\n", key);
        printf("Decrypted message: \"%s\"\n", out);
        /* no critical section because worst case scenario the variable is just
          written multiple times */
        found = 1;
      }
    }
    free(out);
  }

  const double elapsed = omp_get_wtime() - tstart;
  printf("Elapsed time: %f\n", elapsed);
  assert(found); /* ensure that we did found the key */
  return EXIT_SUCCESS;
}
