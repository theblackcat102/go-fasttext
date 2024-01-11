#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void *FastTextHandle;

FastTextHandle NewHandle(const char *path);
void DeleteHandle(FastTextHandle handle);
// these prototypes are pretty badly defined, 
// don't ask me why, asked the original author
char *Predict(FastTextHandle handle, char *query);
char *Analogy(FastTextHandle handle, char *, char *, char *, int32_t);
char *Wordvec(FastTextHandle handle, char *query);
char *Neighbor(FastTextHandle handle, char *query, int32_t k);

#ifdef __cplusplus
}
#endif
