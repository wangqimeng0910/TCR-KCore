
/*

Repair -- an implementation of Larsson and Moffat's compression and
decompression algorithms.
Copyright (C) 2010-current_year Gonzalo Navarro

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

Author's contact: Gonzalo Navarro, Dept. of Computer Science, University of
Chile. Blanco Encalada 2120, Santiago, Chile. gnavarro@dcc.uchile.cl

*/

// linear probing hash table for pairs

// table only grows to keep factor, does not shrink
// this is ok for this application where #pairs rarely decreases

// value -1 denotes empty cells, -2 is a deletion mark

#include "hash.h"

#include <stdlib.h>

extern float factor;  // load factor

typedef unsigned long long relong;
#define LPRIME ((relong)767865341467865341)
#define PRIME 2013686449

int searchHash(Thash H, Tpair p)  // returns id

{
  // relong: unsigned long long

  // 将 p.left 和 p.right 合并成一个 relong 类型的数, 高32位是left 低32位是right
  relong u = ((relong)p.left) << (8 * sizeof(int)) | (relong)p.right;
  
  // 将乘法得出的64位数的表示，右移32位，再取低32位，再与 H.maxpos 进行与运算
  // maxpos 是 1<<x - 1的形式，所以这一步的运算结果，一定是小于等于 maxpos 的，所以table[k]不会越界
  int k = ((LPRIME * u) >> (8 * sizeof(int))) & H.maxpos;
  Trecord *recs = H.Rec->records;
  while (H.table[k] != -1) {
    if ((H.table[k] >= 0) && (recs[H.table[k]].pair.left == p.left) &&
        (recs[H.table[k]].pair.right == p.right))
      break;
    k = (k + 1) & H.maxpos;
  }
  return H.table[k];
}

void deleteHash(Thash *H, int id)  // deletes H->Rec[id].pair from hash

{
  Trecord *rec = H->Rec->records;
  H->table[rec[id].kpos] = -2;
  H->used--;
}

Thash createHash(int maxpos, Trarray *Rec)
// creates new empty hash table

{
  // maxpos 好像是自己设定的一个参数 现在是 256 * 256
  Thash H;
  int i;
  // upgrade maxpos to the next value of the form (1<<smth)-1
  while (maxpos & (maxpos - 1)) maxpos &= maxpos - 1;
  maxpos = (maxpos - 1) << 1 | 1;  // avoids overflow if maxpos = 1<<31
  // maxpos 是 不小于 maxpos中最接近的 2的幂 - 1的数
  H.maxpos = maxpos;
  H.used = 0;
  // table 是 长度为 maxpos + 1 的整形数组
  H.table = (int *)malloc((1 + maxpos) * sizeof(int));
  // 将 table 数组中的每个元素都设为 -1
  for (i = 0; i <= maxpos; i++) H.table[i] = -1;
  H.Rec = Rec;
  return H;
}

static int finsertHash(Thash H, Tpair p)
// inserts w/o resizing, assumes there is space
// does not update used field
// note can reuse marked deletions

{
  relong u = ((relong)p.left) << (8 * sizeof(int)) | (relong)p.right;
  int k = ((LPRIME * u) >> (8 * sizeof(int))) & H.maxpos;
  while (H.table[k] >= 0) k = (k + 1) & H.maxpos;
  return k;
}

void insertHash(Thash *H, int id)  // inserts H->Rec[id].pair in hash
                                   // assumes key is not present
                                   // sets ptr from Rec to hash as well

{
  int k;
  Trecord *rec = H->Rec->records;
  if (H->used > H->maxpos * factor)  // resize
  {
    Thash newH = createHash((H->maxpos << 1) | 1, H->Rec);
    int i;
    int *tab = H->table;
    for (i = 0; i <= H->maxpos; i++)
      if (tab[i] >= 0)  // also removes marked deletions
      {
        k = finsertHash(newH, rec[tab[i]].pair);
        newH.table[k] = tab[i];
        rec[tab[i]].kpos = k;
      }
    newH.used = H->used;
    free(H->table);
    *H = newH;
  }
  H->used++;
  k = finsertHash(*H, rec[id].pair);
  H->table[k] = id;
  rec[id].kpos = k;
}

void destroyHash(Thash *H)

{
  free(H->table);
  H->table = NULL;
  H->maxpos = 0;
  H->used = 0;
}

void hashRepos(Thash *H, int id)

{
  Trecord *rec = H->Rec->records;
  H->table[rec[id].kpos] = id;
}
